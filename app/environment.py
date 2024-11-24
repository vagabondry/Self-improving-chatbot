import os
import json
import numpy as np
import pandas as pd
import torch
import gym
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from tqdm import tqdm
from torch.utils.data import Dataset
from new_train_dataset import format_conversation

# Load model and tokenizer
model_path = "D:/Downloads/conversation-gpt2-with-emotions"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
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
                (pd.to_datetime(fb["timestamp"]) - pd.to_datetime(user_message["timestamp"])).total_seconds()
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
            state_text = f"{user_input} [SEP] {bot_reply} [SEP] {reaction}"
            self.state = self.tokenizer(state_text, return_tensors="pt")
            return self.state

    def step(self, action):
        self.reset()
        user_input = self.state.split("[SEP]")[0]
        # print(user_input)
        # user_input = self.state 
        responses = generate_responses_with_emotions(user_input, self.emotions, self.model, self.tokenizer)
        selected_emotion, selected_response = responses[action]
        # reward = self.calculate_reward(self.state.split("[SEP]")[2], analyze_sentiment(selected_response))  # Combine with feedback if available
        reward = self.calculate_reward("good", analyze_sentiment(selected_response))  # Combine with feedback if available
        return selected_emotion, selected_response, reward
    
    def calculate_reward(self, feedback, sentiment_score):
        # Assign reward based on explicit feedback and inferred sentiment
        reward = 0

        # Explicit feedback
        if feedback.lower() == "good":
            reward += 1
        elif feedback.lower() == "bad":
            reward -= 1

        reward += sentiment_score

        return reward

## Agent ##
class ChatbotAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995):
        self.env = env
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def get_state_key(self, state):
        # Convert state to a hashable key for the Q-table
        return tuple(state.numpy())

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        state_key = self.get_state_key(state)
        return np.argmax(self.q_table.get(state_key, [0] * self.env.action_space.n))  # Exploit

    def train(self, episodes=100):
        for episode in range(episodes):
            # Use the current index for training on sequential JSON entries
            for i in range(len(self.env.conversation_data)):
                state = self.env.reset(index=i)
                total_reward = 0

                while True:
                    state_key = self.get_state_key(state)
                    if state_key not in self.q_table:
                        self.q_table[state_key] = [0] * self.env.action_space.n  # Initialize Q-values

                    action = self.choose_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    total_reward += reward

                    # Update Q-value using Q-learning
                    next_state_key = self.get_state_key(next_state)
                    future_rewards = max(self.q_table.get(next_state_key, [0] * self.env.action_space.n))
                    self.q_table[state_key][action] += self.alpha * (
                        reward + self.gamma * future_rewards - self.q_table[state_key][action]
                    )

                    state = next_state

                    if done:
                        break

            # Decay exploration rate
            self.epsilon = max(0.1, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}/{episodes} | Total Reward: {total_reward}")

### Main Execution ###

if __name__ == "__main__":
    chat_monitor = ChatLogMonitor()
    feedback_monitor = FeedbackLogMonitor()
    env = ChatbotEnv(model, tokenizer, emotions, chat_monitor, feedback_monitor)
    # Add RL Agent and Training Logic
    agent = ChatbotAgent(env)
    agent.train(episodes=10)
