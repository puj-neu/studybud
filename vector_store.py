from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
import pinecone
from config import LOCAL_VECTOR_STORE_DIR, RETRIEVER_K
import shutil

class VectorStore:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def clear_local_store(self):
        """Clear the local vector store directory"""
        try:
            if LOCAL_VECTOR_STORE_DIR.exists():
                shutil.rmtree(LOCAL_VECTOR_STORE_DIR)
            LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to clear vector store: {str(e)}")

    def create_local_store(self, texts):
        # Clear existing vector store before creating new one
        # self.clear_local_store()
        # print("cleared storage")
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        print("Vector done")
        vectordb.persist()
        return vectordb.as_retriever(search_kwargs={'k': RETRIEVER_K})

    def create_pinecone_store(self, texts, api_key: str, environment: str, index_name: str):
        pinecone.init(api_key=api_key, environment=environment)
        vectordb = Pinecone.from_documents(
            documents=texts,
            embedding=self.embeddings,
            index_name=index_name
        )
        return vectordb.as_retriever(search_kwargs={'k': RETRIEVER_K})
