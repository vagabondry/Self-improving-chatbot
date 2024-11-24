# Self-Improving Chatbot

This project is a Telegram chatbot powered by GPT-2, fine-tuned for conversational purposes with added emotional responses. The bot will use deep learning model for language understanding and response generation based on the tone of conversation, and leverage reinforcement learning for improvement based on user feedback. Over time, the chatbot feasibly should refine its responses.

### Features

### Project Structure:
```bash
├── app/
│   ├── bot.py             # Main script for Telegram bot 
│   ├── environment.py     # Reinforcement Learning Environment and Agent
│   ├── new_train_data.py  # Script to preprocess and format new training data (optional)
├── model/
│   └── conversation-gpt2-with-emotions/  # Pre-trained and fine-tuned model files 
├── dataset/
│   └── casual_data_windows.csv  # Dataset used for fine-tuning GPT-2
├── training/
│   └── train_conv.ipynb   # Jupyter notebook for model training (optional)
├── .env                   # Environment variables (ignored by Git)
├── .gitignore             # Ignore sensitive files and unnecessary build artifacts
├── Dockerfile             # Docker configuration for containerizing the app
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Setup and Installation
**Prerequisites**

    Python 3.9 or later
    Docker (for containerized deployment)
    Telegram Bot API token
    Hugging Face API key 

**Step 1**: Clone the Repository
  ```bash
  git clone https://github.com/vagabondry/Self-improving-chatbot.git
  cd Self-improving-chatbot
  ```

**Step 2**: Install Dependencies

  Install Python dependencies using ```pip```:
  ```bash
  pip install -r requirements.txt
  ```
**Step 3**: Configure Environment Variables

Create a .env file in the project root with the following keys:
  ```bash
  TELEGRAM_BOT_TOKEN=your-telegram-bot-token
  HF_API_KEY=your-huggingface-api-key
  ```
**Step 4**: Run the Bot Locally

  Start the bot by running:
  ```bash
  python app/bot.py
  ```

### Environment Variables
Variable	| Description
--- | ---
TELEGRAM_BOT_TOKEN	| Token for authenticating with the Telegram API.
HF_API_KEY	| API key for Hugging Face inference.

To avoid exposing sensitive credentials, these variables are stored in a ```.env``` file, which is excluded from version control.

### Docker Deployment
**Step 1**: Build the Docker Image

  Build the Docker image from the project directory:
  ```bash
  docker build -t telegram-chatbot .
  ```

**Step 2**: Run the Docker Container

  Run the container, passing the environment variables:
  ```bash
  docker run -p 8080:8080 --env-file .env telegram-chatbot
  ```

## License
This project is licensed under the [MIT License](LICENSE).
