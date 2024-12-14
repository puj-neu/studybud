import streamlit as st
import json
from models import Flashcards
from document_processor import DocumentProcessor
from vector_store import VectorStore
from chat_engine import ChatEngine
from flashcard_generator import FlashcardGeneratorOpenAI
import logging
import time
from learning_tools import QuizGenerator,LessonPlanGenerator
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be first Streamlit command
st.set_page_config(page_title="RAG Flashcards")

def add_custom_css():
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            transition: all 0.3s;
        }
        .main-header {
            color: #2E4053;
            text-align: center;
            padding: 20px;
            background: #F0E6E6;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .flashcard-header {
            color: #1A5276;
            padding: 10px;
        }
        .css-1d391kg {  /* Sidebar styling */
        background-color: #f7e6ff;
        padding: 2rem 1rem;
        border-right: 1px solid #e6ccff;
        }
        .stButton>button {  /* Button styling */
            background-color: #e6f3ff;
            color: #2c3e50;
            border: 1px solid #b3d9ff;
        }
        .stButton>button:hover {  /* Button hover effect */
            background-color: #ccebff;
            border-color: #99d6ff;
        }       
            </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'flashcards' not in st.session_state:
    st.session_state.flashcards = Flashcards([])
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'lesson_plan_data' not in st.session_state:
    st.session_state.lesson_plan_data = None
if 'quiz_data' not in st.session_state:
    st.session_state.quiz_data = None
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'flashcards_generated' not in st.session_state:
    st.session_state.flashcards_generated = False


def input_fields():
    with st.sidebar:
        st.image("img/logostudybuddy.jpg", width=300)
        st.markdown("### 🔑 Configuration")
        st.session_state.openai_api_key = st.secrets.get("openai_api_key", "") or st.text_input(
            "OpenAI API key", 
            type="password", 
            key="openai_key_input"
        )
        
        st.markdown("### 📄 Document Upload")
        st.session_state.source_docs = st.file_uploader(
            "Upload Documents",
            type="pdf",
            accept_multiple_files=True,
            key="doc_uploader"
        )
        if st.button("🚀 Process Documents", key="process_button"):
            st.session_state.flashcards_generated = False
            process_documents()
    

def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.source_docs:
        st.warning("⚠️ Please provide OpenAI API key and upload documents.")
        return
    
    try:
        with st.spinner('🔄 Processing documents...'):
            doc_processor = DocumentProcessor()
            doc_processor.save_uploaded_files(st.session_state.source_docs)
            documents = doc_processor.load_documents()
            texts = doc_processor.split_documents(documents)
            
            vector_store = VectorStore(st.session_state.openai_api_key)
            st.session_state.retriever = vector_store.create_local_store(texts)
            
            doc_processor.cleanup_temp_files()
            st.success("✅ Documents processed successfully!")
            
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

def generate_flashcards():
    if st.session_state.retriever is None:
        st.warning("⚠️ Please process documents first.")
        return
        
    # Clear existing flashcards
    st.session_state.flashcards.data.clear()
    
    with st.spinner("🔄 Generating flashcards..."):
        generator = FlashcardGeneratorOpenAI(api_key=st.session_state.openai_api_key)
        docx = st.session_state.retriever.get_relevant_documents("")
        # Track unique content
        generated_content = set()
        flashcard_count = 0
        max_flashcards = 5
        
        # Process documents until we have enough flashcards
        for doc in docx:
            if flashcard_count >= max_flashcards:
                break
                
            try:
                content = doc.page_content[:200].strip()
                if content and content not in generated_content:
                    flashcard = generator.generate_flashcard(content)
                    if flashcard and flashcard.input_expression:  # Verify valid flashcard
                        st.session_state.flashcards.data.append(flashcard)
                        generated_content.add(content)
                        flashcard_count += 1
            except Exception as e:
                logger.error(f"Error generating flashcard: {str(e)}")
                continue
        
        st.session_state.flashcards_generated = True
        if flashcard_count > 0:
            st.success(f"✅ Generated {flashcard_count} unique flashcards!")
        else:
            st.warning("⚠️ Could not generate any valid flashcards.")

def show_flashcards():
    if len(st.session_state.flashcards.data) == 0:
        st.info("💡 Generate flashcards by processing documents first")
    else:
        st.markdown("<h3 class='flashcard-header'>📝 Your Flashcards</h3>", unsafe_allow_html=True)
        for flashcard in st.session_state.flashcards.data:
            with st.expander(f"🔍 {flashcard.input_expression}", expanded=False):
                st.markdown(f"**Answer:** {flashcard.output_expression}")
                st.markdown(f"**Example:** ✨ {flashcard.example_usage}")

def display_lesson_plan(lesson_plan_data: dict):
    if lesson_plan_data:
        st.markdown("### 📅 7-Day Study Plan")
        
        # Format each day as a separate bullet point
        days = lesson_plan_data['week_plan'].split('Day')
        for i, day in enumerate(days[1:], 1):  # Skip empty first split
            # Clean up the day's text and split into sentences
            day_text = day.strip().split('. ')
            # Remove any empty sentences and join with bullet points
            day_sentences = [s.strip() for s in day_text if s.strip()]
            
            # Display the day header
            st.markdown(f"**Day {i}:**")
            # Display each sentence as a bullet point
            for sentence in day_sentences:
                if sentence:  # Only display non-empty sentences
                    st.markdown(f"- {sentence}")
            st.markdown("")  # Add spacing between days
        
        # Display other sections
        st.markdown("### 📚 Main Topics")
        st.markdown(f"```{lesson_plan_data['topics']}```")
        
        st.markdown("### 🔍 Additional Resources")
        st.markdown(f"```{lesson_plan_data['resources']}```")

def display_quiz(quiz_data: dict):
    if quiz_data:
        st.write("### 📝 Practice Quiz")
        for idx, (q, a, d) in enumerate(zip(
            quiz_data["questions"],
            quiz_data["answers"],
            quiz_data["difficulty"]
        )):
            st.write(f"**Question {idx+1}** ({d}) 📌")
            st.write(q)
            
            # Create a unique key for each question's answer button
            button_key = f"answer_btn_{idx}"
            if st.button("Show Answer", key=button_key):
                st.session_state.quiz_answers[idx] = True
            
            # Display answer if it has been revealed
            if st.session_state.quiz_answers.get(idx, False):
                st.write(f"**Answer:** ✨ {a}")
            
            st.divider()


def main():
    add_custom_css()
    
    st.markdown("<h1 class='main-header'>📚 STUDY BUDDY </h1>", unsafe_allow_html=True)
    
    input_fields()
    
    # Process Documents button
    # col2 = st.columns([3, 1])
    # with col2:
    #     if st.button("🚀 Process Documents", key="process_button"):
    #         process_documents()
    
    # Chat interface
    # st.markdown("### 💬 Chat Interface")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            st.chat_message("human", avatar="🧑").write(message[0])
            st.chat_message("assistant", avatar="🤖").write(message[1])
    
    if query := st.chat_input("Ask a question about your documents", key="chat_input"):
        st.chat_message("human", avatar="🧑").write(query)
        if st.session_state.retriever is not None:
            chat_engine = ChatEngine(st.session_state.openai_api_key)
            chain = chat_engine.create_chain(st.session_state.retriever)
            response = chain.invoke(query)
            st.chat_message("assistant", avatar="🤖").write(response)
            st.session_state.chat_history.append((query, response))
        else:
            st.warning("⚠️ Please process documents first.")
    
    # Show content tabs if documents are processed
    if st.session_state.retriever is not None:
    # Initialize tab-specific session states if not exists
        if 'active_tab' not in st.session_state:
            st.session_state.active_tab = "Flashcards"
        if 'generating_flashcards' not in st.session_state:
            st.session_state.generating_flashcards = False
        if 'generating_lesson_plan' not in st.session_state:
            st.session_state.generating_lesson_plan = False
        if 'generating_quiz' not in st.session_state:
            st.session_state.generating_quiz = False

        tab1, tab2, tab3 = st.tabs(["📇 Flashcards", "📅 Lesson Plan", "📝 Quiz"])
        
        with tab1:
            if not st.session_state.flashcards_generated:
                if st.button("✨ Generate Flashcards", key="flashcard_button"):
                    st.session_state.generating_flashcards = True
                    st.session_state.active_tab = "Flashcards"
                    generate_flashcards()
                    st.experimental_rerun()
            show_flashcards()
        
        with tab2:
            if not st.session_state.lesson_plan_data:
                if st.button("✨ Generate Lesson Plan", key="plan_button"):
                    st.session_state.generating_lesson_plan = True
                    st.session_state.active_tab = "Lesson Plan"
                    with st.spinner("🔄 Creating your personalized lesson plan..."):
                        planner = LessonPlanGenerator(st.session_state.openai_api_key)
                        documents = st.session_state.retriever.get_relevant_documents("")
                        content = "\n".join([doc.page_content for doc in documents])
                        st.session_state.lesson_plan_data = planner.generate_plan(content)
                    st.experimental_rerun()
            if st.session_state.lesson_plan_data:
                display_lesson_plan(st.session_state.lesson_plan_data)
        
        with tab3:
            if not st.session_state.quiz_data:
                if st.button("🎯 Generate Quiz", key="quiz_button"):
                    st.session_state.generating_quiz = True
                    st.session_state.active_tab = "Quiz"
                    with st.spinner("🔄 Creating your practice quiz..."):
                        quiz_gen = QuizGenerator(st.session_state.openai_api_key)
                        documents = st.session_state.retriever.get_relevant_documents("")
                        content = "\n".join([doc.page_content for doc in documents])
                        st.session_state.quiz_data = quiz_gen.generate_quiz(content)
                        st.session_state.quiz_answers = {}
                    st.experimental_rerun()
            if st.session_state.quiz_data:
                display_quiz(st.session_state.quiz_data)


if __name__ == '__main__':
    main()
