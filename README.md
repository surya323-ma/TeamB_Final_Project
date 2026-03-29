# BotTrainer — Local LLM NLU Studio (Ollama)

Intent classification and entity extraction using **Ollama** (100% local, no API key needed).

## Models Used
- Text prediction: `llama3`
- Image prediction: `llava`

## Setup

### 1. Install Ollama
Download from https://ollama.com and install.

### 2. Pull the models
```bash
ollama pull llama3
ollama pull llava
```

### 3. Start Ollama server
```bash
ollama serve
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure (optional)
```bash
cp .env.example .env
# Edit .env to change OLLAMA_URL or model names if needed
```

### 6. Run the app
```bash
streamlit run app.py
```

## Input Modes
- Text — type your message
- Voice — record audio (transcribed via Google Speech)
- Image — upload screenshot/photo (analyzed by llava)

## Project Structure
```
bottrainer/
├── app.py                  # Streamlit UI
├── intent_prompt.txt       # Prompt template reference
├── intents.json            # Add/edit your intents here
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml         # Hides Streamlit auto page nav
└── src/
    ├── nlu_pipeline.py     # Ollama: llama3 (text) + llava (image)
    └── evaluator.py        # sklearn metrics
```
