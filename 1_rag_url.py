import os
from dotenv import load_dotenv
load_dotenv()

import bs4
import requests
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

def debug_webpage_content(url):
    """Debug function to check raw webpage content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = bs4.BeautifulSoup(response.text, 'html.parser')
        
        print("\nDebug Information:")
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)} bytes")
        
        # Look for common content containers
        content = None
        content_selectors = [
            'article',
            'main',
            '.post-content',
            '.article-content',
            '.entry-content',
            '.content',
            '#content',
            '.document',
            '.body'
        ]
        
        for selector in content_selectors:
            if selector.startswith('.'):
                element = soup.find(class_=selector[1:])
            elif selector.startswith('#'):
                element = soup.find(id=selector[1:])
            else:
                element = soup.find(selector)
                
            if element:
                # Convert to text and clean up
                content = element.get_text(separator=' ', strip=True)
                print(f"\nFound content using selector: {selector}")
                break
        
        if not content:
            # Fallback: try to get content from the body
            body = soup.find('body')
            if body:
                # Remove script and style elements
                for element in body.find_all(['script', 'style', 'nav', 'header', 'footer']):
                    element.decompose()
                content = body.get_text(separator=' ', strip=True)
                print("\nUsing fallback content extraction from body")
        
        if content:
            # Clean up the text
            content = ' '.join(content.split())  # Normalize whitespace
            print("\nContent preview:")
            print(content[:500] + "...")
            return content
        else:
            raise ValueError("No content extracted from webpage")
        
    except Exception as e:
        print(f"Error in debug_webpage_content: {str(e)}")
        return None

def create_rag_chain(url):
    """Create a RAG chain for a given URL"""
    print(f"\nAttempting to create RAG chain with URL: {url}")
    
    try:
        # 1. Load content using our debug_webpage_content
        print(f"\nLoading content from {url}...")
        content = debug_webpage_content(url)
        
        if not content:
            raise ValueError("No content extracted from webpage")
            
        # Create a document from the extracted content
        docs = [Document(page_content=content, metadata={"source": url})]
        
        print(f"Loaded document with {len(content)} characters")
        print("\nContent preview:")
        print(content[:500] + "...")

        # 2. Split documents into chunks
        print("\nSplitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False
        )
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            raise ValueError(f"No splits created from documents. Content length: {len(content)}")
            
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
        llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
        if 'docs' in locals() and docs:
            print(f"Document content preview: {docs[0].page_content[:200]}...")
        raise

def main():
    if not os.getenv("openai_api_key"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")

    print("Creating RAG chain...")
    
    urls = [
        "https://libertystreeteconomics.newyorkfed.org/2024/12/do-import-tariffs-protect-u-s-firms/",
        "https://libertystreeteconomics.newyorkfed.org/2024/11/why-do-banks-fail-bank-runs-versus-solvency/"
    ]
    
    for url in urls:
        try:
            print(f"\nAttempting to create RAG chain with URL: {url}")
            rag_chain, vectorstore = create_rag_chain(url)
            print("Successfully created RAG chain!")
            break
        except Exception as e:
            print(f"Failed with URL {url}: {str(e)}")
            continue
    else:
        raise Exception("Failed to create RAG chain with all URLs")

    try:
        while True:
            question = input("\nEnter your question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break

            # Generate and print answer
            print("\nGenerating answer...")
            response = rag_chain.invoke(question)
            print("\nQuestion:", question)
            print("\nAnswer:", response)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        # Cleanup
        print("\nCleaning up...")
        try:
            vectorstore.delete_collection()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()

# query
# What are the main reasons banks fail?
# How can we prevent bank runs effectively?
# What early warning signs indicate a bank is at risk?