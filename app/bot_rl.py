import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler, filters
import json
from datetime import datetime
import os
import random
import hashlib
import pandas as pd
from environment import ChatbotEnv, ChatLogMonitor, FeedbackLogMonitor, ChatbotAgent


model_path = "kkaterina/conversation-gpt2-with-emotions"

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Load the fine-tuned GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_path)

# If you have a GPU, move the model to the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Initialize the RL environment and agent
emotions = {"neutral", "disapproval", "caring", "annoyance", "anger", "excitement", "joy"}
chat_monitor = ChatLogMonitor()
feedback_monitor = FeedbackLogMonitor()
rl_env = ChatbotEnv(model, tokenizer, emotions, chat_monitor, feedback_monitor)
agent = ChatbotAgent(rl_env)
desired_labels = {'neutral', 'disapproval', 'caring', 'annoyance', 'anger', 'excitement', 'joy'}
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device=device)

# Feedback context to match feedback to messages
feedback_context = {}


def generate_response_via_rl(prompt, env, agent):
    """
    Use the RL environment and agent to generate a response based on the prompt.
    """
    env.pending_user_messages.append({"message": prompt, "timestamp": datetime.now().isoformat()})
    state = env.reset()  # Reset environment with the user prompt

    # Let the agent choose an action based on the RL policy
    action = agent.choose_action(state)
    selected_emotion, response, _ = env.step(action)  # Generate the response
    return response

def generate_response_with_emotion(prompt, model, tokenizer, max_length=30, top_k=50):
    model.eval()
    prompt = f"{prompt}[SEP]"
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=False)

    tokens_to_keep = ["[SEP]"]

    filtered_output = " ".join([
        token for token in decoded_output.split() 
        if token not in tokenizer.all_special_tokens or token in tokens_to_keep
    ])

    return filtered_output.split("[SEP]")[1].strip()

def analyse_sentiment(user_reply, classifier):
    model_outputs = classifier(user_reply)
    model_outputs = model_outputs[0]

    filtered_outputs = [output for output in model_outputs if output['label'] in desired_labels]
    
    relevant_emotions = [output['label'].upper() for output in filtered_outputs if output['score'] > 0.2]
    
    if not relevant_emotions:
        relevant_emotions = ["NEUTRAL"]
    
    # Combine emotion tokens with the formatted conversation
    emotion_tokens = " ".join([f"[{emotion}]" for emotion in relevant_emotions])
    return f"{emotion_tokens} {user_reply}"

def log_message(message_data, chat_id):
    try:
        log_dir = "/app/chat_logs" 
        os.makedirs(log_dir, exist_ok=True)
        file_name = f"{log_dir}/chat_{chat_id}.json"

        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(message_data)
        print(message_data)

        with open(file_name, "w") as file:
            json.dump(logs, file, indent=4)
    except Exception as e:
        print(f"Error logging message for chat {chat_id}: {e}")

def log_feedback(feedback_data, chat_id):
    try:
        feedback_dir = "/app/feedback_logs" 
        os.makedirs(feedback_dir, exist_ok=True)
        file_name = f"{feedback_dir}/chat_{chat_id}_feedback.json"

        if os.path.exists(file_name):
            with open(file_name, "r") as file:
                logs = json.load(file)
        else:
            logs = []

        logs.append(feedback_data)
        print(feedback_data)

        with open(file_name, "w") as file:
            json.dump(logs, file, indent=4)
    except Exception as e:
        print(f"Error logging feedback for chat {chat_id}: {e}")

async def start(update: Update, context):
    if update.message.chat.type == "private":
        await update.message.reply_text("Hi! I'm your AI bot. Send me a message!")
    else:
        await update.message.reply_text("Hi! Add me to your group and mention me to interact.")

async def chat(update: Update, context):
    user_message = update.message.text
    chat_id = update.message.chat.id
    chat_type = update.message.chat.type
    bot_username = context.bot.username

    message_data = {
        "chat_id": chat_id,
        "user_id": update.message.from_user.id,
        "username": update.message.from_user.username,
        "message": user_message,
        "timestamp": datetime.now().isoformat(),
        "chat_type": chat_type,
    }
    log_message(message_data, chat_id)

    if chat_type in ["group", "supergroup"]:
        if f"@{bot_username}" in user_message:
            user_message = user_message.replace(f"@{bot_username}", "").strip()
        else:
            if random.random() > 0.33:
                return

    try:

        response = generate_response_via_rl(user_message, rl_env, agent)
        

        feedback_id = hashlib.md5(f"{chat_id}:{user_message}:{response}".encode()).hexdigest()
        feedback_context[feedback_id] = {
            "user_message": user_message,
            "bot_response": response,
            "chat_id": chat_id,
            "timestamp": datetime.now().isoformat(),
        }

        keyboard = [
            [
                InlineKeyboardButton("Good 👍", callback_data=f"feedback:Good:{feedback_id}"),
                InlineKeyboardButton("Bad 👎", callback_data=f"feedback:Bad:{feedback_id}")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(response, reply_markup=reply_markup)

    except Exception as e:
        await update.message.reply_text(f"An error occurred: {e}")

async def feedback_handler(update: Update, context):
    query = update.callback_query
    await query.answer()

    data = query.data.split(":")
    feedback = data[1]  # Good or Bad
    feedback_id = data[2]  # Feedback ID
    feedback_data = feedback_context.get(feedback_id)

    if feedback_data:
        feedback_entry = {
            "user_id": query.from_user.id,
            "username": query.from_user.username,
            "chat_id": feedback_data["chat_id"],
            "user_message": feedback_data["user_message"],
            "bot_response": feedback_data["bot_response"],
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        log_feedback(feedback_entry, feedback_data["chat_id"])

        await query.edit_message_reply_markup(reply_markup=None)

BOT_TOKEN = "bot_token"

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))
app.add_handler(CallbackQueryHandler(feedback_handler))

if __name__ == "__main__":
    print("Bot is running...")
    app.run_polling()
