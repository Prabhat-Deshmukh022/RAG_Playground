# RAG Architecture Comparison API

This project provides a FastAPI-based backend for uploading PDF documents and comparing different Retrieval-Augmented Generation (RAG) pipelines: **Basic RAG**, **Hybrid RAG**, and **Auto-Merge RAG**. It supports chunking, embedding, vector search, and querying with various LLMs (Gemini, Mistral, Groq, etc.), and is designed for easy experimentation and benchmarking.

---

## Features

- **Upload Multiple PDFs:** Upload up to three PDF files at once for ingestion.
- **Three RAG Pipelines:** Choose between Basic, Hybrid, and Auto-Merge RAG architectures.
- **Flexible LLM Integration:** Easily switch between different LLM providers (Gemini, Mistral, Groq, etc.).
- **Contextual Querying:** Retrieve relevant document chunks and generate answers using the selected LLM.
- **Extensible Design:** Modular codebase for adding new RAG strategies or LLMs.
- **API-First:** Built with FastAPI for easy integration with frontends or other services.

---

## Project Structure

```
RAG_Application/
│
├── backend/
│   ├── main.py                # FastAPI app with endpoints for upload and query
│   ├── rag_implementation.py  # RAG pipeline implementations (Basic, Hybrid, AutoMerge)
│   ├── remove_file_contents.py
│   ├── language_model_api.py
│   └── temp_uploads/          # Temporary storage for uploaded PDFs
│
├── RAG/
│   ├── RAG_Engine.py          # Base RAGEngine interface
│   └── RAG_Interfaces/        # Interface definitions for modularity
│
├── eval_script/
│   └── ...                    # Evaluation scripts (e.g., RAGAS, LangSmith)
│
├── .env                       # Environment variables (API keys, etc.)
└── requirements.txt           # Python dependencies
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd RAG_Application
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- fastapi
- uvicorn
- langchain
- llama-index
- sentence-transformers
- torch
- faiss-cpu
- pypdf
- tenacity
- python-dotenv

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

Add any other keys as needed for your LLM providers.

---

## Running the API Server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## API Endpoints

### 1. Health Check

**GET** `/healthz`

Returns:
```json
{"status": "ok"}
```

---

### 2. Upload PDFs

**POST** `/upload`

**Form Data:**
- `rag_option` (int): 1 = Basic, 2 = Hybrid, 3 = Auto-Merge
- `llm_option` (int): LLM selection (see your `language_model_api.py` for mapping)
- `files`: Up to 3 PDF files

**Example (using curl):**
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "rag_option=1" \
  -F "llm_option=1" \
  -F "files=@yourfile1.pdf" \
  -F "files=@yourfile2.pdf"
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {"filename": "yourfile1.pdf", "status": "ingested"},
    {"filename": "yourfile2.pdf", "status": "ingested"}
  ]
}
```

---

### 3. Query the RAG Pipeline

**POST** `/query`

**Form Data:**
- `option` (int): The RAG pipeline to use (should match the one used in upload)
- `query` (str): Your question

**Example (using curl):**
```bash
curl -X POST "http://127.0.0.1:8000/query?option=1" \
  -F "query=What is the summary of the document?"
```

**Response:**
```json
{
  "status": "success",
  "answer": "The summary is ...",
  "contexts": ["context chunk 1", "context chunk 2", ...]
}
```

---

## RAG Pipelines Explained

- **Basic_RAG:** Uses manual chunking, custom embedding with sentence-transformers, and FAISS for vector search.
- **Hybrid_RAG:** Combines dense (FAISS) and sparse (BM25) retrieval using LangChain's EnsembleRetriever.
- **AutoMerge_RAG:** Uses LlamaIndex's AutoMergingRetriever for advanced context aggregation.

All pipelines support prompt construction and querying via your selected LLM.

---

## Customization

- **Add new LLMs:** Extend `language_model_api.py` and pass new options to the RAG classes.
- **Change chunking/embedding:** Modify the `chunk` and `embed` methods in the RAG classes.
- **Evaluation:** Use scripts in `eval_script/` for RAGAS or LangSmith-based evaluation.

---

## Troubleshooting

- **API Key Errors:** Ensure your `.env` file is present and correct.
- **CUDA/CPU:** By default, embeddings run on CPU. Modify device selection in `rag_implementation.py` if you want GPU.
- **File Limits:** The `/upload` endpoint is set for up to 3 PDFs by default.
- **Rate Limits:** If using cloud LLMs, be aware of API rate limits and quotas.

---

## License

MIT License

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## Contact

For questions or contributions, please open an issue or pull request on the repository.
