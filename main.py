import streamlit as st
import json
from models import Flashcards
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_engine import ChatEngine
from flashcard_generator import FlashcardGeneratorOpenAI


st.set_page_config(page_title="RAG Flashcards")

if 'flashcards' not in st.session_state:
    st.session_state.flashcards = Flashcards([])
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
# Initialize session state variables
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = Flashcards([])
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []



st.title("RAG Flashcards Generator")

def input_fields():
    with st.sidebar:
        st.session_state.openai_api_key = st.secrets.get("openai_api_key", "") or st.text_input("OpenAI API key", type="password", key="openai_key_input")
        st.session_state.input_language = st.selectbox("Input Language", ["English", "Polish"], key="input_lang")
        st.session_state.output_language = st.selectbox("Output Language", ["English", "Polish"], key="output_lang")
        st.session_state.source_docs = st.file_uploader("Upload Documents", type="pdf", accept_multiple_files=True, key="doc_uploader")

import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning("Please provide OpenAI API key and upload documents.")
        return
    
    try:
        with st.spinner('Processing documents...'):
            start_time = time.time()
            
            logger.info("Starting document processing")
            doc_processor = DocumentProcessor()
            
            logger.info("Saving uploaded files")
            doc_processor.save_uploaded_files(st.session_state.source_docs)
            
            logger.info("Loading documents")
            documents = doc_processor.load_documents()
            logger.info(f"Loaded {len(documents)} documents")
            
            logger.info("Splitting documents")
            texts = doc_processor.split_documents(documents)
            logger.info(f"Created {len(texts)} text chunks")
            
            logger.info("Initializing vector store")
            vector_store = VectorStore(st.session_state.openai_api_key)
            
            logger.info("Creating local store and retriever")
            st.session_state.retriever = vector_store.create_local_store(texts)
            
            logger.info("Generating flashcards")
            generator = FlashcardGeneratorOpenAI(api_key=st.session_state.openai_api_key)
            max_flashcards = 5
            
            # Select most relevant chunks for flashcard generation
            selected_texts = texts[:max_flashcards] if len(texts) > max_flashcards else texts
            
            for text in selected_texts:
                try:
                    flashcard = generator.generate_flashcard(text.page_content[:200])
                    st.session_state.flashcards.data.append(flashcard)
                except Exception as e:
                    logger.error(f"Error generating flashcard: {str(e)}")
                    continue
            
            
            logger.info("Cleaning up temporary files")
            doc_processor.cleanup_temp_files()
            
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"Total processing time: {processing_time:.2f} seconds")
            
            st.success(f"Documents processed and {len(st.session_state.flashcards.data)} flashcards generated successfully!")
            
    except Exception as e:
        logger.error(f"Process failed with error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


def show_flashcards():
    if len(st.session_state.flashcards.data) == 0:
        st.info("Generate flashcards by processing documents first")
    else:
        for flashcard in st.session_state.flashcards.data:
            with st.expander(flashcard.input_expression, expanded=False):
                st.write(f"**Translation:** {flashcard.output_expression}")
                st.write(f"**Example:** {flashcard.example_usage}")

def import_export_flashcards():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Import Flashcards", type="json")
        if uploaded_file:
            data = json.load(uploaded_file)
            st.session_state.flashcards = Flashcards.import_from_json(data)
            st.success(f"Imported {len(st.session_state.flashcards.data)} flashcards!")
    
    with col2:
        if len(st.session_state.flashcards.data) > 0:
            st.download_button(
                "Export Flashcards",
                data=json.dumps(st.session_state.flashcards.as_json(), indent=2),
                file_name="flashcards_export.json",
                mime="application/json"
            )
def generate_flashcards():
    if st.session_state.flashcards_pending and st.session_state.retriever:
        with st.spinner('Generating flashcards...'):
            generator = FlashcardGeneratorOpenAI(api_key=st.session_state.openai_api_key)
            for doc in st.session_state.retriever.get_relevant_documents(""):
                flashcard = generator.generate_flashcard(
                    doc.page_content[:200],  # Limit content length
                )
                st.session_state.flashcards.data.append(flashcard)
        st.session_state.flashcards_pending = False


def main():
    input_fields()
    
    if st.button("Process Documents", key="process_button"):
        process_documents()
    
    # Chat interface
    # Display chat history
    for message in st.session_state.chat_history:
        st.chat_message("human").write(message[0])
        st.chat_message("assistant").write(message[1])
    
    # Handle new queries
    if query := st.chat_input("Ask a question about your documents"):
        st.chat_message("human").write(query)
        if st.session_state.retriever is not None:
            chat_engine = ChatEngine(st.session_state.openai_api_key)
            chain = chat_engine.create_chain(st.session_state.retriever)
            response = chain.invoke(query)
            st.chat_message("assistant").write(response)
            st.session_state.chat_history.append((query, response))
        else:
            st.warning("Please process documents first.")
    # # Generate flashcards in background
    # if st.session_state.get('flashcards_pending', False):
    #     generate_flashcards()
    
    # Show flashcards if available
    if len(st.session_state.flashcards.data) > 0:
        show_flashcards()

if __name__ == '__main__':
    main()
