import os
import json
import numpy as np
import torch
import gym
from datetime import datetime
from DQN import DQNAgent
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from tqdm import tqdm
from torch.utils.data import Dataset
from new_train_dataset import format_conversation

# Load model and tokenizer
model_path = "kkaterina/conversation-gpt2-with-emotions"
tokenizer = GPT2Tokenizer.from_pretrained(repo_name=model_path)
model = GPT2LMHeadModel.from_pretrained(repo_name=model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")
emotions = {"neutral", "disapproval", "caring", "annoyance", "anger", "excitement", "joy"}


### Helper Functions ###

def generate_responses_with_emotions(prompt, emotions, model, tokenizer, max_length=30, top_k=50):
    """Generate responses for each emotion."""
    model.eval()
    responses = []
    for emotion in emotions:
        prompt_with_emotion = f"[{emotion.upper()}] {prompt}"
        input_ids = tokenizer.encode(prompt_with_emotion, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                do_sample=True,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                num_return_sequences=1,
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append((emotion, response))
    return responses


def analyze_sentiment(text):
    """Perform sentiment analysis on user replies."""
    sentiment = sentiment_analyzer([text])[0]
    if sentiment["label"] == "POS":
        return sentiment["score"]
    elif sentiment["label"] == "NEG":
        return -sentiment["score"]
    return 0


def match_user_replies_with_responses(chat_logs, feedback_logs):
    """Match user messages from chat_logs with bot responses from feedback_logs."""
    matched_data = []
    for user_message in chat_logs:
        closest_feedback = min(
            feedback_logs,
            key=lambda fb: abs(
                (datetime.fromisoformat(fb["timestamp"]) - datetime.fromisoformat(user_message["timestamp"])).total_seconds()
            ),
        )
        matched_data.append(
            {
                "user_message": user_message["message"],
                "bot_response": closest_feedback["bot_response"],
                "feedback": closest_feedback["feedback"],
            }
        )
    return matched_data

def collect_and_format_data(chat_logs, feedback_logs, classifier):
    """Prepare data for fine-tuning."""
    matched_data = match_user_replies_with_responses(chat_logs, feedback_logs)
    dataset = []
    for interaction in matched_data:
        user_message = interaction["user_message"]
        bot_response = interaction["bot_response"]
        feedback = interaction["feedback"].lower()

        # Exclude negative feedback
        if feedback == "bad":
            continue

        # Format conversation with emotion classification
        formatted_entry = format_conversation({
            "0": user_message,
            "1": bot_response,
            "2": None
        }, classifier)
        dataset.append(formatted_entry)
    return dataset


### Monitors for Logs ###

class ChatLogMonitor:
    """Monitor chat logs for new user messages."""
    def __init__(self, log_folder="chat_logs"):
        self.log_folder = log_folder
        self.seen_files = set()

    def get_new_messages(self):
        all_files = set(os.listdir(self.log_folder))
        new_files = all_files - self.seen_files
        self.seen_files.update(new_files)

        new_messages = []
        for file_name in new_files:
            if file_name.startswith("chat_") and file_name.endswith(".json"):
                with open(os.path.join(self.log_folder, file_name), "r") as file:
                    new_messages.extend(json.load(file))
        return new_messages


class FeedbackLogMonitor:
    """Monitor feedback logs for user responses to bot outputs."""
    def __init__(self, log_folder="feedback_logs"):
        self.log_folder = log_folder
        self.seen_files = set()

    def get_feedback_data(self):
        all_files = set(os.listdir(self.log_folder))
        new_files = all_files - self.seen_files
        self.seen_files.update(new_files)

        feedback_data = []
        for file_name in new_files:
            if file_name.startswith("feedback_") and file_name.endswith(".json"):
                with open(os.path.join(self.log_folder, file_name), "r") as file:
                    feedback_data.extend(json.load(file))
        return feedback_data


### Reinforcement Learning Environment ###

class ChatbotEnv:
    def __init__(self, model, tokenizer, emotions, chat_log_monitor, feedback_log_monitor):
        self.model = model
        self.tokenizer = tokenizer
        self.emotions = emotions
        self.chat_log_monitor = chat_log_monitor
        self.feedback_log_monitor = feedback_log_monitor
        self.state = None
        self.pending_user_messages = []
        self.feedback_data = []
        self.action_space = gym.spaces.Discrete(len(emotions))

    def update_logs(self):
        self.pending_user_messages.extend(self.chat_log_monitor.get_new_messages())
        self.feedback_data.extend(self.feedback_log_monitor.get_feedback_data())

    def reset(self):
        if not self.pending_user_messages:
            self.update_logs()

        if self.pending_user_messages:
            user_message = self.pending_user_messages.pop(0)
            user_input = user_message["message"]

            feedback_entry = next(
                (fb for fb in self.feedback_data if fb["user_message"] == user_input),
                None,
            )
            bot_reply = feedback_entry["bot_response"] if feedback_entry else ""
            reaction = feedback_entry["feedback"] if feedback_entry else "neutral"

            input_ids = self.tokenizer.encode(user_input, return_tensors="pt").to(device)
            reply_ids = self.tokenizer.encode(bot_reply, return_tensors="pt").to(device)

            self.state = torch.cat((input_ids.mean(1), reply_ids.mean(1)), dim=-1).squeeze().cpu().numpy()
            return self.state

    def step(self, action):
        user_input = self.state[:len(self.state) // 2]  
        responses = generate_responses_with_emotions(user_input, self.emotions, self.model, self.tokenizer)

        selected_emotion, selected_response = responses[action]
        sentiment_score = analyze_sentiment(selected_response)
        reward = self.calculate_reward("good", sentiment_score) # dummy calculation

        response_ids = self.tokenizer.encode(selected_response, return_tensors="pt").to(device)
        next_state = torch.cat((user_input, response_ids.mean(1).cpu()), dim=-1).numpy()

        done = True  
        return next_state, reward, done

    def calculate_reward(self, feedback, sentiment_score):
        reward = 0
        if feedback.lower() == "good":
            reward += 1
        elif feedback.lower() == "bad":
            reward -= 1
        reward += sentiment_score
        return reward


### Main Execution ###

if __name__ == "__main__":
    chat_monitor = ChatLogMonitor()
    feedback_monitor = FeedbackLogMonitor()
    env = ChatbotEnv(model, tokenizer, emotions, chat_monitor, feedback_monitor)
    # Add RL Agent and Training Logic
    state_size = 512  
    action_size = len(emotions)
    agent = DQNAgent(state_size, action_size)
    agent.train(env=env, agent=agent, episodes=10)
