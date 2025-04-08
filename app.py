import gradio as gr
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import tempfile
import shutil
import git
import glob

# Load environment variables
load_dotenv()

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the language model (using a small, efficient model)
llm = HuggingFaceHub(
    repo_id="google/flan-t5-small",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

def clone_repo(repo_url, branch="main"):
    """Clone the GitHub repository to a temporary directory"""
    temp_dir = tempfile.mkdtemp()
    git.Repo.clone_from(repo_url, temp_dir, branch=branch)
    return temp_dir

def load_documents(repo_path):
    """Load documents from the repository"""
    documents = []
    
    # Define the base path for financial reports
    base_path = os.path.join(repo_path, "financial-reports", "companies")
    
    # Get all company folders
    company_folders = glob.glob(os.path.join(base_path, "*"))
    
    for company_folder in company_folders:
        # Get all PDF files in the company folder
        pdf_files = glob.glob(os.path.join(company_folder, "*.pdf"))
        
        for pdf_file in pdf_files:
            try:
                # Load PDF file
                loader = PyPDFLoader(pdf_file)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {pdf_file}: {str(e)}")
    
    return documents

def create_vector_store(documents):
    """Create a vector store from the documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def setup_rag():
    """Setup the RAG system"""
    # Clone the repository
    repo_url = os.getenv("GITHUB_REPO_URL")
    repo_path = clone_repo(repo_url)
    
    # Load and process documents
    documents = load_documents(repo_path)
    if not documents:
        raise ValueError("No documents found in the repository. Please check the folder structure and file formats.")
    
    vector_store = create_vector_store(documents)
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3})
    )
    
    # Clean up
    shutil.rmtree(repo_path)
    
    return qa_chain

def answer_question(question, qa_chain):
    """Answer questions using the RAG system"""
    try:
        response = qa_chain.invoke({"query": question})
        return response["result"]
    except Exception as e:
        return f"Error: {str(e)}"

# Initialize the RAG system
qa_chain = setup_rag()

# Create the Gradio interface
def gradio_interface(question):
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
