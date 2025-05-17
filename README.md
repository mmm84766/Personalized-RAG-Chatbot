# Personalized RAG Chatbot

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/mmm84766/Personalized-RAG-Chatbot)

A powerful chatbot that combines Retrieval-Augmented Generation (RAG) with personalized responses based on user preferences. The chatbot uses OpenRouter for LLM access and local embeddings for document processing.

## Features

### Personalization Settings
- **Tone**: Friendly, Professional, Casual, Academic
- **Communication Goal**: Educate, Entertain, Inform, Persuade
- **Response Length**: Brief, Detailed, Comprehensive
- **Response Style**: Storytelling, Technical, Conversational, Formal
- **Language Preference**: Multiple language support
- **User Persona**: Beginner, Expert, Young Learner

### Document Processing
- Support for PDF and TXT files
- Automatic chunking and embedding
- Vector-based retrieval using Chroma DB
- Local embeddings using SentenceTransformer

### Chat Interface
- Real-time chat with context-aware responses
- Dynamic personalization settings
- Conversation history management
- Document upload and processing

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/mmm84766/Personalized-RAG-Chatbot.git
   cd Personalized-RAG-Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenRouter API**
   - Sign up for an account at [OpenRouter](https://openrouter.ai/)
   - Get your API key from the dashboard
   - Modify a `.env` file in the project root with the following content:
     ```
     OPENROUTER_API_KEY=your_api_key_here
     OPENROUTER_API_BASE=https://openrouter.ai/api/v1
     MODEL_NAME=mistralai/mistral-7b-instruct
     SITE_URL=http://localhost:8501
     SITE_NAME=Personalized RAG Chatbot
     ```

## Usage

1. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`

3. **Configure the chatbot**
   - Select a persona or customize settings in the sidebar
   - Upload documents for context
   - Start chatting!

## Architecture Overview

### Key Components

1. **app.py**: Streamlit UI application
   - User interface
   - Document upload handling
   - Chat interaction

2. **config.py**: Configuration settings
   - Personalization options
   - Default settings
   - Persona configurations

3. **rag_engine.py**: RAG implementation
   - Document processing
   - Vector storage
   - Local embeddings using SentenceTransformer

4. **chatbot.py**: Chatbot logic
   - OpenRouter integration
   - Response generation
   - Conversation management

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Chroma DB
- SentenceTransformer
- PyPDF
- python-dotenv

## Notes

- The application uses local embeddings for document processing to avoid API costs
- OpenRouter provides access to various LLM models
- Default model is set to Mistral-7B-Instruct, but can be changed in the `.env` file
- Document processing is done locally for better privacy and cost efficiency


## Sample Documents

The repository includes a sample document (`sample_docs/ai_introduction.txt`) for testing the RAG functionality.

## Architecture

The application is built with the following components:

- `app.py`: Streamlit UI and main application logic
- `config.py`: Configuration and personalization settings
- `rag_engine.py`: Document processing and retrieval engine
- `chatbot.py`: Chatbot logic and response generation

