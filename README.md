# RAG Architecture Comparison Application

A full-stack application for uploading PDFs, ingesting them into different Retrieval-Augmented Generation (RAG) pipelines, and querying them using various LLMs (Gemini, Mistral, Groq, etc.). The project features a FastAPI backend and a Next.js (React) frontend.

---

## Features

- **Upload up to 3 PDFs** at once for ingestion.
- **Choose between three RAG architectures:** Basic RAG, Hybrid RAG, and Auto-Merge RAG.
- **Select LLM provider:** Gemini, Mistral, or Groq.
- **Chat interface** for querying ingested documents.
- **Context display:** See which document chunks were used to answer your question.
- **Extensible backend:** Modular design for adding new RAG strategies or LLMs.
- **Evaluation scripts** for benchmarking (RAGAS, LangSmith, etc.).

---

## Project Structure

```
RAG_Application/
│
├── backend/
│   ├── main.py                  # FastAPI app (API endpoints)
│   ├── rag_implementation.py    # RAG pipeline implementations
│   ├── remove_file_contents.py  # Utility to clear temp files
│   ├── language_model_api.py    # LLM API abstraction
│   └── temp_uploads/            # Temporary PDF storage
│
├── RAG/
│   ├── RAG_Engine.py            # RAGEngine interface
│   └── RAG_Interfaces/          # Modular interfaces (Chunker, Embedding, etc.)
│
├── eval_script/
│   └── ...                      # Evaluation scripts (RAGAS, LangSmith, etc.)
│
├── frontend/
│   └── src/
│       └── app/
│           └── query/
│               └── page.tsx     # Main chat/query page
│       └── components/          # UI components (FormattedLLMResponse, ContextToggleButton, etc.)
│
├── .env                         # Environment variables (API keys, etc.)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Backend Setup (FastAPI)

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

### 4. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
```

### 5. Run the API Server

```bash
cd backend
uvicorn main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Backend API Endpoints

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
- `llm_option` (int): 1 = Gemini, 2 = Mistral, 3 = Groq
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

**POST** `/query?option=<rag_option>`

**Form Data:**
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

## Frontend Setup (Next.js/React)

### 1. Install Node.js dependencies

```bash
cd frontend
npm install
```

### 2. Set API URL

Create a `.env.local` file in `frontend/`:

```
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### 3. Run the Frontend

```bash
npm run dev
```

The frontend will be available at [http://localhost:3000](http://localhost:3000).

---

## Frontend Usage

- **Upload PDFs:** Use the upload section to select up to 3 PDFs, choose RAG architecture and LLM, and upload.
- **Chat:** Enter your question in the chat box and press Enter or click Send.
- **Context Display:** For each bot answer, click the context button to view the document chunks used for the answer.
- **Switch RAG/LLM:** Use the dropdowns to change RAG architecture or LLM for new uploads.

---

## RAG Pipelines Explained

- **Basic RAG:** Manual chunking, custom embedding (sentence-transformers), FAISS vector search.
- **Hybrid RAG:** Combines dense (FAISS) and sparse (BM25) retrieval using LangChain's EnsembleRetriever.
- **Auto-Merge RAG:** Uses LlamaIndex's AutoMergingRetriever for advanced context aggregation.

---

## Customization

- **Add new LLMs:** Extend `language_model_api.py` and update the frontend dropdown.
- **Change chunking/embedding:** Modify the respective methods in `rag_implementation.py`.
- **Evaluation:** Use scripts in `eval_script/` for RAGAS or LangSmith-based evaluation.

---

## Troubleshooting

- **API Key Errors:** Ensure your `.env` file is present and correct.
- **CUDA/CPU:** By default, embeddings run on CPU. Modify device selection in `rag_implementation.py` if you want GPU.
- **File Limits:** The `/upload` endpoint is set for up to 3 PDFs by default.
- **Rate Limits:** If using cloud LLMs, be aware of API rate limits and quotas.
- **CORS Issues:** The backend enables CORS for all origins by default. Restrict in production.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LlamaIndex](https://github.com/jerryjliu/llama_index)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Next.js](https://nextjs.org/)

---
