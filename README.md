# Financial Reports RAG Application

This is a Retrieval-Augmented Generation (RAG) application that allows users to ask questions about financial reports stored in a GitHub repository. The application uses a lightweight model (flan-t5-small) and efficient embeddings to provide answers based on the content of the financial reports.

## Features

- Connects to a GitHub repository containing financial reports
- Uses efficient sentence transformers for embeddings
- Implements RAG using FAISS for vector storage
- Provides a simple Gradio interface for user interaction
- Runs efficiently on 2 vCPU and 16GB RAM

## Setup Instructions

1. Create a new repository on GitHub to store your financial reports
2. Create a new Space on Hugging Face:
   - Go to huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Select "Python" as the runtime

3. Set up environment variables in your Hugging Face Space:
   - Go to your Space settings
   - Add the following secrets:
     - `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token
     - `GITHUB_REPO_URL`: The URL of your GitHub repository containing financial reports

4. Upload the files to your Hugging Face Space:
   - `app.py`
   - `requirements.txt`

## GitHub Repository Setup

1. Create a new repository on GitHub
2. Create a folder structure like this:
   ```
   financial-reports/
   ├── company1/
   │   ├── 2023_annual_report.txt
   │   └── 2023_quarterly_reports/
   ├── company2/
   │   ├── 2023_annual_report.txt
   │   └── 2023_quarterly_reports/
   ```

3. Add your financial reports as text files in the appropriate folders

## Usage

1. Once deployed, visit your Hugging Face Space URL
2. Enter your question in the text box
3. The application will search through the financial reports and provide an answer based on the content

## Technical Details

- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Language Model: mistralai/Mistral-7B-Instruct-v0.1
- Vector Store: FAISS
- Framework: LangChain
- UI: Gradio

## Resource Requirements

- CPU: 2 vCPU
- RAM: 16GB
- Storage: Depends on the size of your financial reports

## Limitations

- The application works best with text-based financial reports
- The model has a context window limitation, so very long documents are chunked 

## Demo

Model is hosted on hugging face you can [try the same](https://huggingface.co/spaces/Risov/financial-rag-app)
