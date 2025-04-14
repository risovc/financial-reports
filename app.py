import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
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
import time
import math # For ceiling division for batches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Global Variables ---
embeddings_model = None
llm_model = None
qa_chain_global = None
initialization_status = "Initializing..."
init_error_message = None

# --- Initialization Functions ---

def initialize_models():
    """Initialize the embedding and language models"""
    global embeddings_model, llm_model, initialization_status, init_error_message
    logger.info("Attempting to initialize models...")
    start_time = time.time()
    try:
        logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} # Or 'cuda'
        )
        logger.info("Embedding model loaded successfully.")

        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN environment variable is not set")

        logger.info("Loading LLM model endpoint: mistralai/Mistral-7B-Instruct-v0.1")
        llm_model = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            huggingfacehub_api_token=api_token,
            task="text-generation",
            temperature=0.7,
            max_new_tokens=512,
        )
        logger.info("LLM model endpoint loaded successfully.")

        elapsed = time.time() - start_time
        logger.info(f"Model initialization completed in {elapsed:.2f} seconds.")
        return True

    except Exception as e:
        init_error_message = f"Model initialization failed: {str(e)}"
        logger.error(init_error_message, exc_info=True)
        initialization_status = f"Error: {init_error_message}"
        embeddings_model = None
        llm_model = None
        return False

def clone_repo(repo_url, branch="main"):
    """Clone the GitHub repository to a temporary directory"""
    global init_error_message, initialization_status
    logger.info(f"Attempting to clone repository: {repo_url} (branch: {branch})")
    start_time = time.time()
    try:
        temp_dir = tempfile.mkdtemp()
        logger.info(f"Cloning into temporary directory: {temp_dir}")
        git.Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
        elapsed = time.time() - start_time
        logger.info(f"Repository cloned successfully in {elapsed:.2f} seconds.")
        return temp_dir
    except Exception as e:
        init_error_message = f"Repository cloning failed: {str(e)}"
        logger.error(init_error_message, exc_info=True)
        initialization_status = f"Error: {init_error_message}"
        return None

def load_documents(repo_path):
    """Load PDF documents with progress reporting"""
    global init_error_message, initialization_status
    logger.info("Attempting to load documents from repository path...")
    start_time = time.time()
    documents = []
    total_pages_loaded = 0 # Track total pages

    base_path = os.path.join(repo_path, "companies")
    logger.info(f"Scanning for PDF documents in subdirectories of: {base_path}")

    if not os.path.exists(base_path):
        init_error_message = f"Required directory structure not found: {base_path}. Expected structure: ./companies/[company_name]/report.pdf"
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return None, 0 # Return None for documents and 0 pages

    pdf_files_found = glob.glob(os.path.join(base_path, "*", "*.pdf"), recursive=True)
    total_files = len(pdf_files_found)
    logger.info(f"Found {total_files} potential PDF files.")

    if not pdf_files_found:
        init_error_message = "No PDF files found within the 'companies' subdirectories."
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return None, 0

    loaded_count = 0
    failed_count = 0
    # Define progress reporting interval (e.g., report every 5 files or every 20%)
    report_interval = max(1, total_files // 5) # Report at least 5 times + start/end

    for i, pdf_file in enumerate(pdf_files_found):
        try:
            # Log individual file loading start
            # logger.info(f"Loading PDF [{i+1}/{total_files}]: {os.path.basename(pdf_file)}") # Can be verbose
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            if docs:
                num_pages = len(docs)
                documents.extend(docs)
                total_pages_loaded += num_pages
                # Log success per file (optional, can make logs long)
                # logger.info(f"Successfully loaded {num_pages} pages from {os.path.basename(pdf_file)}")
                loaded_count += 1
            else:
                logger.warning(f"No content loaded from {pdf_file}")
                failed_count += 1

            # Report progress periodically
            if (i + 1) % report_interval == 0 or (i + 1) == total_files:
                 logger.info(f"Loading Progress: Processed {i+1}/{total_files} files. Total pages loaded so far: {total_pages_loaded}")

        except Exception as e:
            logger.error(f"Error loading {pdf_file}: {str(e)}", exc_info=False)
            failed_count += 1

    elapsed = time.time() - start_time
    logger.info(f"Document loading completed in {elapsed:.2f} seconds.")
    logger.info(f"Successfully loaded {loaded_count} files, failed to load {failed_count} files.")

    if not documents:
        init_error_message = "No documents were successfully loaded from any PDF files."
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return None, 0
    else:
        logger.info(f"Total document pages loaded: {total_pages_loaded}")
        return documents, total_pages_loaded

def create_vector_store_batched(documents, embeddings, batch_size=100):
    """Create FAISS vector store by adding documents in batches with progress"""
    global init_error_message, initialization_status
    if not documents:
        init_error_message = "Cannot create vector store: No documents were provided."
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return None

    logger.info("Step 1: Splitting documents into text chunks...")
    start_split_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    total_chunks = len(texts)
    split_elapsed = time.time() - start_split_time
    logger.info(f"Split {len(documents)} document pages into {total_chunks} text chunks in {split_elapsed:.2f}s.")

    if not texts:
        init_error_message = "Splitting documents resulted in zero text chunks."
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return None

    logger.info(f"Step 2: Creating FAISS vector store in batches (batch size: {batch_size})...")
    start_index_time = time.time()
    vector_store = None
    total_batches = math.ceil(total_chunks / batch_size)

    try:
        for i in range(0, total_chunks, batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = texts[i : i + batch_size]
            batch_start_chunk = i + 1
            batch_end_chunk = min(i + batch_size, total_chunks)

            logger.info(f"Processing Batch {batch_num}/{total_batches} (Chunks {batch_start_chunk}-{batch_end_chunk} / {total_chunks})...")
            batch_time_start = time.time()

            if vector_store is None:
                # Initialize the store with the first batch
                vector_store = FAISS.from_documents(batch_docs, embeddings)
                logger.info(f"   Initialized vector store with first batch.")
            else:
                # Add subsequent batches to the existing store
                vector_store.add_documents(batch_docs)
                logger.info(f"   Added batch {batch_num} to vector store.")

            batch_time_elapsed = time.time() - batch_time_start
            logger.info(f"   Batch {batch_num} processed in {batch_time_elapsed:.2f}s.")

        index_elapsed = time.time() - start_index_time
        logger.info(f"Vector store creation completed in {index_elapsed:.2f} seconds.")
        return vector_store

    except Exception as e:
        init_error_message = f"Vector store creation failed during batch processing: {str(e)}"
        logger.error(init_error_message, exc_info=True)
        initialization_status = f"Error: {init_error_message}"
        return None

def setup_rag_pipeline():
    """Sets up the complete RAG pipeline with enhanced logging"""
    global qa_chain_global, initialization_status, init_error_message, embeddings_model, llm_model
    logger.info("--- Starting RAG Pipeline Setup ---")
    overall_start_time = time.time()
    initialization_status = "Setting up RAG Pipeline..."

    # 1. Initialize Models
    initialization_status = "Initializing embedding and LLM models..."
    if embeddings_model is None or llm_model is None:
        if not initialize_models():
            return False

    # 2. Get Repo URL
    repo_url = os.getenv("GITHUB_REPO_URL")
    repo_branch = os.getenv("GITHUB_REPO_BRANCH", "main")
    if not repo_url:
        init_error_message = "GITHUB_REPO_URL environment variable is not set."
        logger.error(init_error_message)
        initialization_status = f"Error: {init_error_message}"
        return False

    # 3. Clone Repo
    initialization_status = "Cloning repository..."
    repo_path_temp = clone_repo(repo_url, repo_branch)
    if repo_path_temp is None:
         return False

    # 4. Load Documents
    initialization_status = f"Loading documents from {repo_path_temp}..."
    documents_loaded, pages_count = load_documents(repo_path_temp) # Get documents and page count
    if documents_loaded is None:
        shutil.rmtree(repo_path_temp)
        logger.warning(f"Cleaned up temporary directory due to loading error: {repo_path_temp}")
        return False

    # 5. Create Vector Store (Batched)
    initialization_status = "Creating vector store (chunking & embedding)..."
    vector_store_created = create_vector_store_batched(documents_loaded, embeddings_model, batch_size=256) # Adjust batch_size as needed
    if vector_store_created is None:
        shutil.rmtree(repo_path_temp)
        logger.warning(f"Cleaned up temporary directory due to vector store error: {repo_path_temp}")
        return False

    # 6. Create QA Chain
    initialization_status = "Creating retrieval chain..."
    logger.info("Creating RetrievalQA chain...")
    start_time = time.time()
    try:
        qa_chain_global = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=vector_store_created.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=False
        )
        elapsed = time.time() - start_time
        logger.info(f"RetrievalQA chain created successfully in {elapsed:.2f} seconds.")

    except Exception as e:
        init_error_message = f"Failed to create RetrievalQA chain: {str(e)}"
        logger.error(init_error_message, exc_info=True)
        initialization_status = f"Error: {init_error_message}"
        shutil.rmtree(repo_path_temp)
        logger.warning(f"Cleaned up temporary directory due to chain creation error: {repo_path_temp}")
        qa_chain_global = None
        return False

    # 7. Cleanup
    logger.info(f"Cleaning up temporary repository directory: {repo_path_temp}")
    try:
        shutil.rmtree(repo_path_temp)
        logger.info("Temporary directory removed.")
    except Exception as e:
        logger.error(f"Error removing temporary directory {repo_path_temp}: {str(e)}", exc_info=True)

    overall_elapsed = time.time() - overall_start_time
    initialization_status = "Ready"
    logger.info(f"--- RAG Pipeline Setup Completed Successfully in {overall_elapsed:.2f} seconds ---")
    init_error_message = None
    return True

# --- Gradio Interface Logic ---

def answer_question_interface(question):
    """Gradio interface function"""
    if qa_chain_global is None:
        logger.error("answer_question_interface called but qa_chain is not initialized.")
        return f"Error: The RAG pipeline failed to initialize. Please check the logs. Last known error: {init_error_message or 'Unknown initialization error'}"

    if not question or question.strip() == "":
        return "Please enter a question."

    logger.info(f"Received question: {question}")
    try:
        response = qa_chain_global.invoke({"query": question})
        result = response.get("result", "No result found in the response.")
        logger.info(f"Generated answer: {result[:100]}...")
        return result
    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return f"An error occurred while trying to answer the question. Details: {str(e)}"

# --- Perform Setup ---
setup_successful = setup_rag_pipeline()

# --- Build Gradio UI ---
logger.info("Building Gradio interface...")

sample_prompts = [
    "What were the total revenues reported?",
    "Summarize the key financial highlights.",
    "What are the main risks mentioned in the reports?",
    "Is there any information about future outlook or guidance?",
    "What company names are mentioned in the documents?",
    "Compare the performance metrics if multiple reports are available.",
]

with gr.Blocks() as iface:
    gr.Markdown("# Financial Reports Q&A Bot")
    gr.Markdown(
        "Ask questions about financial data contained in the PDF reports from the configured GitHub repository. "
        "The system uses Retrieval-Augmented Generation (RAG) to find relevant information and generate answers."
    )

    with gr.Row():
        status_text = gr.Textbox(
            label="System Status",
            value=initialization_status, # Shows final status after setup completes
            interactive=False,
            max_lines=2
        )

    with gr.Column():
        question_input = gr.Textbox(
            lines=3,
            placeholder="Type your question here...",
            label="Your Question",
            interactive=setup_successful
        )
        answer_output = gr.Textbox(
            lines=8,
            label="Answer",
            interactive=False
        )
        submit_button = gr.Button("Ask Question", variant="primary", interactive=setup_successful)

    gr.Examples(
        examples=sample_prompts,
        inputs=question_input,
        label="Sample Questions (click to use)"
    )

    submit_button.click(
        fn=answer_question_interface,
        inputs=question_input,
        outputs=answer_output
    )
    question_input.submit(
         fn=answer_question_interface,
        inputs=question_input,
        outputs=answer_output
    )

    if not setup_successful:
        gr.Markdown(
            f"**Initialization Failed:** The application could not start correctly. "
            f"Please check the console logs for details. "
            f"Last error: `{init_error_message}`"
        )

logger.info("Gradio interface built.")

# --- Launch App ---
if __name__ == "__main__":
    logger.info("Launching Gradio interface...")
    iface.launch()
    logger.info("Gradio interface launched. Access it via the URL provided.")
