import tempfile
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from config import TMP_DIR, CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    @staticmethod
    def load_documents():
        loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
        return loader.load()

    @staticmethod
    def split_documents(documents):
        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        return text_splitter.split_documents(documents)

    @staticmethod
    def save_uploaded_files(files):
        for file in files:
            with tempfile.NamedTemporaryFile(
                delete=False, 
                dir=TMP_DIR.as_posix(), 
                suffix='.pdf'
            ) as tmp_file:
                tmp_file.write(file.read())

    @staticmethod
    def cleanup_temp_files():
        for file in TMP_DIR.iterdir():
            file.unlink()
