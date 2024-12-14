# Retrieval Augmented Generation Engine with LangChain, Streamlit, and Vector DB + LLM Fine Tuning and Prompt Engineering
LINK TO VIDEO - https://northeastern-my.sharepoint.com/:v:/g/personal/chikkavadaragudipr_m_northeastern_edu/EaRGplvZyelIqbLkwXxVSwIB25Gt3hx5IXnlIU4Mktvq6A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D&e=O4FAI5
## Project Summary

This Retrieval Augmented Generation (RAG) Engine combines LangChain, Streamlit, and Pinecone to create a robust web application for document analysis, summarization, and interactive Q&A. Users can upload PDFs, generate vector embeddings from the text, and engage in conversational interactions with the content. The system maintains chat history for a more dynamic experience.

## Key Capabilities

- **Web Interface**: Built on Streamlit for user-friendly interaction
- **Credential Management**: Secure input for OpenAI and Pinecone API keys
- **Multi-PDF Upload**: Process multiple documents simultaneously
- **Text Chunking**: Divide documents into manageable segments for model compatibility
- **Embedding Generation**: Convert text chunks to vector embeddings
- **Storage Options**: Choose between Pinecone or local vector storage
- **Conversational AI**: Engage in Q&A sessions with uploaded documents, with persistent chat history
- **Learning Tools**: Generate flashcards, lesson plans, and quizzes from documents

## System Requirements

- Python 3.7 or newer
- LangChain library
- Streamlit framework
- Valid OpenAI API key
- PDF documents for analysis

## Setup and Operation

1. Clone the repository:
   ```bash
   git clone https://github.com/puj-neu/studybud.git
   cd studybud
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```bash
   streamlit run main.py
   ```

4. Open the provided URL in a web browser

5. Enter API credentials via the sidebar or in src/.streamlit/secrets.toml

6. Upload PDF documents for processing

7. Click "Submit Documents" to process and embed the content

8. Start asking questions to interact with the documents

9. Use additional features to create flashcards, lesson plans, and quizzes

## Fine Tuning dataset - 

1. https://www.kaggle.com/datasets/fernandosr85/khan-academy-exercises

## Project Team

Manish CP, Keerthi Nethiguntla, Puja Kalivarapu

