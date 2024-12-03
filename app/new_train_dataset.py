import pandas as pd
from transformers import pipeline
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler

model_path = "kkaterina/conversation-gpt2-with-emotions"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Tokenize the dataset
def tokenize_dataset(dataset, tokenizer):
    return [
        tokenizer(entry, truncation=True, padding="max_length", return_tensors="pt")
        for entry in dataset
    ]

# Fine-tune the model
def fine_tune_model(dataset, tokenizer, model):
    dataloader = DataLoader(tokenize_dataset(dataset, tokenizer), batch_size=4, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * 3)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

# Define desired emotion labels
desired_labels = {'neutral', 'disapproval', 'caring', 'annoyance', 'anger', 'excitement', 'joy'}

# Initialize the emotion classifier
classifier = pipeline(
    task="text-classification", 
    model="SamLowe/roberta-base-go_emotions", 
    top_k=None, 
    device=device
)

def format_conversation(row, classifier):
    """
    Formats a conversation row into a string with emotion labels based on classification.

    Args:
        row (pd.Series): A row from the DataFrame containing conversation parts.
        classifier (pipeline): The emotion classification pipeline.

    Returns:
        str: A formatted string with emotion labels and conversation text.
    """
    # Construct the conversation string
    if pd.notna(row['2']):
        formatted = f"{row['0']} [SEP] {row['1']} [SEP] {row['2']}"
    else:
        formatted = f"{row['0']} [SEP] {row['1']}"

    # Get emotion classification outputs
    model_outputs = classifier(formatted)
    model_outputs = model_outputs[0]  # Extract classification results

    # Filter outputs for desired emotion labels
    filtered_outputs = [
        output for output in model_outputs 
        if output['label'] in desired_labels
    ]
    
    # Extract labels with scores > 0.2
    relevant_emotions = [
        output['label'].upper() 
        for output in filtered_outputs 
        if output['score'] > 0.2
    ]
    
    # Default to NEUTRAL if no emotion scores meet the threshold
    if not relevant_emotions:
        relevant_emotions = ["NEUTRAL"]
    
    # Combine emotion tokens with the formatted conversation
    emotion_tokens = " ".join([f"[{emotion}]" for emotion in relevant_emotions])
    return f"{emotion_tokens} {formatted}"

class ConversationDataset(Dataset):
    """
    A PyTorch Dataset for tokenized conversations.

    Args:
        conversations (list): List of formatted conversation strings.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        max_length (int): Maximum sequence length for tokenization.
    """
    def __init__(self, conversations, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.conversations = conversations
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        text = self.conversations[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0)  
        }


if __name__ == "__main__":
    # Add a formatted text column
    df['formatted_text'] = df.apply(lambda row: format_conversation(row, classifier), axis=1)


