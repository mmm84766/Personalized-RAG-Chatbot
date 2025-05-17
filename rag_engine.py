import os
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from config import CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION_NAME

# Load environment variables
load_dotenv()

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embedding class for SentenceTransformer models."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the SentenceTransformer model."""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using the SentenceTransformer model."""
        embedding = self.model.encode(text)
        return embedding.tolist()

class RAGEngine:
    def __init__(self, use_openai: bool = False):
        """
        Initialize the RAG engine.
        
        Args:
            use_openai (bool): Whether to use OpenAI embeddings (True) or local embeddings (False)
        """
        self.use_openai = use_openai
        if use_openai:
            self.embeddings = OpenAIEmbeddings(
                openai_api_base=os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                model="openai/text-embedding-ada-002",
                default_headers={
                    "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:8501"),
                    "X-Title": os.getenv("SITE_NAME", "Personalized RAG Chatbot")
                }
            )
        else:
            # Use SentenceTransformer with custom embedding class
            self.embeddings = SentenceTransformerEmbeddings()
        
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def load_document(self, file_path: str) -> List[str]:
        """
        Load and parse a document (PDF or TXT).
        
        Args:
            file_path (str): Path to the document
            
        Returns:
            List[str]: List of document chunks
        """
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file format. Please use PDF or TXT files.")
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def process_documents(self, file_paths: List[str]) -> None:
        """
        Process multiple documents and store them in the vector database.
        
        Args:
            file_paths (List[str]): List of paths to documents
        """
        all_chunks = []
        for file_path in file_paths:
            chunks = self.load_document(file_path)
            all_chunks.extend(chunks)
        
        # Create or update the vector store
        self.vector_store = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME
        )

    def retrieve_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant chunks from the vector store for a given query.
        
        Args:
            query (str): The search query
            k (int): Number of chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        if not self.vector_store:
            raise ValueError("No documents have been processed yet.")
        
        results = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

    def clear_documents(self) -> None:
        """Clear all documents from the vector store."""
        if self.vector_store:
            self.vector_store.delete_collection()
            self.vector_store = None 