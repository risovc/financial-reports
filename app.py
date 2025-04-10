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

# Global variable to store initialization error
INIT_ERROR = None

def initialize_models():
    """Initialize the embedding and language models"""
    global INIT_ERROR
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")

        # --- CHANGE THE REPO_ID and potentially remove task/kwargs ---
        llm = HuggingFaceEndpoint(
            # repo_id="google/flan-t5-large",  # Example: Use Flan-T5
            repo_id="mistralai/Mistral-7B-Instruct-v0.1", # Example: Use Mistral Instruct
            huggingfacehub_api_token=api_token,
            task="text-generation", # Often inferred, but can be set
            temperature=0.7,         # Adjust temperature as needed
            max_new_tokens=512,      # Use max_new_tokens for generative models
            # Remove or adjust model_kwargs based on the new model's capabilities
            # You might not need any specific model_kwargs initially
            # model_kwargs={ } 
        )
        # --- END CHANGE ---

        return embeddings, llm
    except Exception as e:
        INIT_ERROR = f"Model initialization failed: {str(e)}"
        logger.error(INIT_ERROR)
        return None, None

def clone_repo(repo_url, branch="main"):
    """Clone the GitHub repository to a temporary directory"""
    global INIT_ERROR
    try:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Cloning repository {repo_url} to {temp_dir}")
        git.Repo.clone_from(repo_url, temp_dir, branch=branch)
        return temp_dir
    except Exception as e:
        INIT_ERROR = f"Repository cloning failed: {str(e)}"
        logger.error(INIT_ERROR)
        raise

def load_documents(repo_path):
    """Load documents from the repository"""
    global INIT_ERROR
    documents = []
    
    # Define the base path for financial reports - updated to match actual structure
    base_path = os.path.join(repo_path, "companies")
    logger.info(f"Looking for documents in: {base_path}")
    
    if not os.path.exists(base_path):
        INIT_ERROR = f"Directory structure not found: {base_path}"
        logger.error(INIT_ERROR)
        return documents
    
    # Get all company folders
    company_folders = glob.glob(os.path.join(base_path, "*"))
    logger.info(f"Found {len(company_folders)} company folders")
    
    if not company_folders:
        INIT_ERROR = "No company folders found in the repository"
        logger.error(INIT_ERROR)
        return documents
    
    for company_folder in company_folders:
        company_name = os.path.basename(company_folder)
        logger.info(f"Processing company folder: {company_name}")
        
        # Get all PDF files in the company folder
        pdf_files = glob.glob(os.path.join(company_folder, "*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {company_name}")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {company_name}")
            continue
            
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
    
    if not documents:
        INIT_ERROR = "No documents were successfully loaded from any PDF files"
        logger.error(INIT_ERROR)
    else:
        logger.info(f"Total documents loaded: {len(documents)}")
    return documents

def create_vector_store(documents):
    """Create a vector store from the documents"""
    global INIT_ERROR
    if not documents:
        INIT_ERROR = "No documents available to create vector store"
        logger.error(INIT_ERROR)
        return None
        
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"Created {len(texts)} text chunks")
        
        vector_store = FAISS.from_documents(texts, embeddings)
        logger.info("Vector store created successfully")
        return vector_store
    except Exception as e:
        INIT_ERROR = f"Vector store creation failed: {str(e)}"
        logger.error(INIT_ERROR)
        return None

def setup_rag():
    """Setup the RAG system"""
    global INIT_ERROR
    try:
        # Check for repository URL
        repo_url = os.getenv("GITHUB_REPO_URL")
        if not repo_url:
            INIT_ERROR = "GITHUB_REPO_URL environment variable is not set"
            logger.error(INIT_ERROR)
            return None
            
        repo_path = clone_repo(repo_url)
        
        # Load and process documents
        documents = load_documents(repo_path)
        if not documents:
            return None
        
        vector_store = create_vector_store(documents)
        if not vector_store:
            return None
        
        # Create the QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3})
        )
        logger.info("RAG system setup completed successfully")
        INIT_ERROR = None  # Clear any previous errors
        
        # Clean up
        shutil.rmtree(repo_path)
        
        return qa_chain
    except Exception as e:
        INIT_ERROR = f"RAG system setup failed: {str(e)}"
        logger.error(INIT_ERROR)
        return None

def answer_question(question, qa_chain):
    """Answer questions using the RAG system"""
    try:
        response = qa_chain.invoke({"query": question})
        return response["result"]
    except Exception as e:
        error_msg = f"Error answering question: {str(e)}"
        logger.error(error_msg)
        return error_msg

# Initialize models
embeddings, llm = initialize_models()
if embeddings is None or llm is None:
    qa_chain = None
else:
    # Initialize the RAG system
    qa_chain = setup_rag()

# Create the Gradio interface
def gradio_interface(question):
    if qa_chain is None:
        return f"Error: RAG system is not properly initialized. {INIT_ERROR}"
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
