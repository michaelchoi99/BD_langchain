import os
from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def create_rag_chain(pdf_path):
    """Create a RAG chain for a given PDF file"""
    print(f"\nAttempting to create RAG chain with PDF: {pdf_path}")
    
    try:
        # 1. Load PDF content
        print(f"Loading PDF from {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            raise ValueError("No content extracted from PDF")
            
        print(f"Loaded {len(pages)} pages")
        print("\nContent preview from first page:")
        print(pages[0].page_content[:500] + "...")

        # 2. Split documents into chunks
        print("\nSplitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(pages)
        
        if not splits:
            raise ValueError(f"No splits created from documents")
            
        print(f"Created {len(splits)} splits")

        # 3. Create embeddings and store in vector database
        print("\nCreating embeddings...")
        embedding = OpenAIEmbeddings()
        
        collection_name = "rag_collection_" + os.urandom(4).hex()
        print(f"Creating Chroma collection: {collection_name}")
        
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name=collection_name
        )
        retriever = vectorstore.as_retriever()

        # 4. Create prompt template from LangChain hub
        print("Loading prompt template...")
        prompt = hub.pull("rlm/rag-prompt")

        # 5. Create LLM
        print("Initializing LLM...")
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        # 6. Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        print("Creating RAG chain...")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain, vectorstore

    except Exception as e:
        print(f"Error in create_rag_chain: {str(e)}")
        raise

def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    print("Creating RAG chain...")
    
    pdf_path = "./fomcminutes20241107.pdf"
    
    try:
        print(f"\nAttempting to create RAG chain with PDF: {pdf_path}")
        rag_chain, vectorstore = create_rag_chain(pdf_path)
        print("Successfully created RAG chain!")
        
        # If we get here, we have a working chain
        try:
            while True:
                question = input("\nEnter your question (or 'quit' to exit): ")
                if question.lower() == 'quit':
                    break

                print("\nGenerating answer...")
                response = rag_chain.invoke(question)
                print("\nQuestion:", question)
                print("\nAnswer:", response)

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            print("\nCleaning up...")
            try:
                vectorstore.delete_collection()
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")
        
    except Exception as e:
        print(f"Failed with PDF {pdf_path}: {str(e)}")
        raise Exception("Failed to create RAG chain")
    
if __name__ == "__main__":
    main()

# query
# What factors influenced the FOMC's decision to lower the federal funds rate by 25 basis points?
# How does the FOMC view the current balance between risks to inflation and employment goals?
# What economic indicators will guide future adjustments to monetary policy?