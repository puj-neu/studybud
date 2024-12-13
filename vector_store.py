from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone
import pinecone
from config import LOCAL_VECTOR_STORE_DIR, RETRIEVER_K

class VectorStore:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    def create_local_store(self, texts):
        vectordb = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
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
