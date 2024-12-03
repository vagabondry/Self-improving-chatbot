FROM python:3.9-slim

# Set a working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

# Expose the port
EXPOSE 8080

# Run your bot
CMD ["python", "app/bot_rl.py"]
