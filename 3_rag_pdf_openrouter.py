import os
from dotenv import load_dotenv
load_dotenv()

import logging
import chromadb

from pathlib import Path
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaRAG:
    def __init__(self, pdf_path, model_name="anthropic/claude-3-sonnet", api_key=None):
        """
        Initialize RAG system with ChromaDB and OpenRouter
        
        Args:
            pdf_path: Path to the PDF file
            model_name: OpenRouter model to use (default: claude-3-sonnet)
            api_key: OpenRouter API key
        """
        self.pdf_path = pdf_path
        self.model_name = model_name
        
        # Initialize OpenRouter client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("openrouter_api_key"),
            default_headers={
                "HTTP-Referer": "https://github.com/michaelchoi99/BD_langchain",
                "X-Title": "PDF RAG System"
            }
        )
        
        # Create a unique collection name based on the PDF filename
        collection_name = f"pdf_collection_{os.path.basename(pdf_path).replace('.', '_')}"
        
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Delete existing collection if it exists
        try:
            self.chroma_client.delete_collection(collection_name)
        except ValueError:
            # Collection doesn't exist
            pass
            
        # Create new collection
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def query(self, question: str, n_results: int = 3) -> dict:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            n_results: Number of chunks to retrieve
            
        Returns:
            dict: Contains answer and source information
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[question],
                n_results=n_results,
                include=["documents", "metadatas"]
            )
            
            # Prepare context from retrieved chunks
            context = []
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                context.append(f"{doc} (Source: Page {metadata['page']})")
            
            full_context = " ".join(context)
            
            # Generate response using OpenRouter
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI assistant that answers questions using only the "
                            "provided context. Include relevant quotes and page numbers in your answer. "
                            f"\n\nContext: {full_context}"
                        )
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )
            
            return {
                "answer": response.choices[0].message.content,
                "sources": results["metadatas"][0]  # Return source metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            raise

    # Keep the load_and_process_document method unchanged
    def load_and_process_document(self):
        """Load and chunk the PDF document"""
        logger.info("Loading PDF document...")
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        logger.info("Splitting document into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(self.chunks)} chunks")
        
        # Prepare chunks and metadata for ChromaDB
        documents = []
        metadatas = []
        for chunk in self.chunks:
            documents.append(chunk.page_content)
            metadatas.append({
                "page": chunk.metadata.get("page", 0),
                "source": self.pdf_path
            })
        
        # Upsert into ChromaDB
        logger.info("Upserting chunks into ChromaDB...")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=[f"chunk_{i}" for i in range(len(documents))]
        )

def main():
    pdf_path = "./fomcminutes20241107.pdf"
    
    try:
        print("\nInitializing RAG system...")
        print("1. Loading PDF document...")
        print("2. Creating vector store...\n")
        
        rag = ChromaRAG(pdf_path, model_name="anthropic/claude-3-sonnet")
        rag.load_and_process_document()
        
        questions = [
            "What are the main topics covered in this document?", # example question
            "Can you summarize the key findings?", # example question
            "What factors influenced the FOMC's decision to lower the federal funds rate by 25 basis points?",
            "How does the FOMC view the current balance between risks to inflation and employment goals?",
            "What economic indicators will guide future adjustments to monetary policy?"
        ]
        
        # Query the system
        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                response = rag.query(question)
                print("\nAnswer:", response["answer"])
                print("\nSources:")
                for source in response["sources"]:
                    print(f"- Page {source['page']}")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")

if __name__ == "__main__":
    main()