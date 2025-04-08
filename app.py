import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile
import shutil
import git
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the language model (using a small, efficient model)
llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-small",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    task="text2text-generation",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

def clone_repo(repo_url, branch="main"):
    """Clone the GitHub repository to a temporary directory"""
    try:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Cloning repository {repo_url} to {temp_dir}")
        git.Repo.clone_from(repo_url, temp_dir, branch=branch)
        return temp_dir
    except Exception as e:
        logger.error(f"Error cloning repository: {str(e)}")
        raise

def load_documents(repo_path):
    """Load documents from the repository"""
    documents = []
    
    # Define the base path for financial reports
    base_path = os.path.join(repo_path, "financial-reports", "companies")
    logger.info(f"Looking for documents in: {base_path}")
    
    if not os.path.exists(base_path):
        logger.error(f"Base path does not exist: {base_path}")
        return documents
    
    # Get all company folders
    company_folders = glob.glob(os.path.join(base_path, "*"))
    logger.info(f"Found {len(company_folders)} company folders")
    
    for company_folder in company_folders:
        company_name = os.path.basename(company_folder)
        logger.info(f"Processing company folder: {company_name}")
        
        # Get all PDF files in the company folder
        pdf_files = glob.glob(os.path.join(company_folder, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {company_name}")
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Loading PDF file: {pdf_file}")
                # Load PDF file
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} pages from {pdf_file}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {str(e)}")
    
    logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def create_vector_store(documents):
    """Create a vector store from the documents"""
    if not documents:
        logger.error("No documents to create vector store")
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    logger.info(f"Created {len(texts)} text chunks")
    
    vector_store = FAISS.from_documents(texts, embeddings)
    logger.info("Vector store created successfully")
    return vector_store

def setup_rag():
    """Setup the RAG system"""
    try:
        # Clone the repository
        repo_url = os.getenv("GITHUB_REPO_URL")
        if not repo_url:
            raise ValueError("GITHUB_REPO_URL environment variable is not set")
            
        repo_path = clone_repo(repo_url)
        
        # Load and process documents
        documents = load_documents(repo_path)
        if not documents:
            raise ValueError("No documents found in the repository. Please check the folder structure and file formats.")
        
        vector_store = create_vector_store(documents)
        if not vector_store:
            raise ValueError("Failed to create vector store")
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        logger.info("RAG system setup completed successfully")
        
        # Clean up
        shutil.rmtree(repo_path)
        
        return qa_chain
    except Exception as e:
        logger.error(f"Error in setup_rag: {str(e)}")
        raise

def answer_question(question, qa_chain):
    """Answer questions using the RAG system"""
    try:
        response = qa_chain.invoke({"query": question})
        return response["result"]
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return f"Error: {str(e)}"

# Initialize the RAG system
try:
    qa_chain = setup_rag()
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {str(e)}")
    qa_chain = None

# Create the Gradio interface
def gradio_interface(question):
    if qa_chain is None:
        return "Error: RAG system is not properly initialized. Please check the logs for details."
    return answer_question(question, qa_chain)

# Launch the interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about the financial reports..."),
    outputs=gr.Textbox(lines=5),
    title="Financial Reports Q&A",
    description="Ask questions about the financial reports stored in the GitHub repository."
)

if __name__ == "__main__":
    iface.launch() 
