# Telegram PDF QA Bot (ChromaDB)

Usage:

- Set environment variable `TELEGRAM_TOKEN` (or create a `.env` with it).
- Install dependencies: `pip install -r requirements.txt`.
- Run: `python bot.py`.

Optional LLM synthesis with Ollama (HTTP API):

- If you have Ollama running locally (default HTTP API at `http://localhost:11434`) and a `llama3` model available, you can enable the optional synthesis step.
- Set `USE_OLLAMA=true` in your `.env` and optionally set `OLLAMA_MODEL` to the model name.
- The bot calls the Ollama HTTP API to generate concise answers based only on the retrieved document passages. This implementation uses `requests` to avoid dependency conflicts.

How it works:

- Upload PDF documents to the bot (as Documents).
- The bot extracts text, chunks it, embeds with `sentence-transformers`, and stores vectors in ChromaDB (in-memory).
- Ask questions; the bot retrieves relevant chunks and returns excerpts. If nothing relevant is found, it replies "Not found in uploaded documents.".

Notes:

- This project calls the Ollama HTTP API when `USE_OLLAMA` is enabled; the host must be running Ollama and exposing the HTTP server. Installing and running Ollama is outside the scope of this repo.
- Recommended workflow: run inside a virtualenv to avoid system package conflicts.

**How It Works (Notes)**

- **Upload**: The bot accepts PDF, PPTX and DOCX files as Telegram Documents and saves them to the project working directory.
- **Extraction**: Text is extracted by `pdf_utils.py` (PDF via PyPDF2, PPTX via python-pptx, DOCX via python-docx).
- **Chunking**: Extracted text is split into overlapping chunks (`chunk_text`) to keep retrieval context manageable.
- **Embeddings**: The app prefers remote embeddings (OpenAI) if `OPENAI_API_KEY` is set. If not available it will attempt to use a local `sentence-transformers` model (requires installing `sentence-transformers` on a compatible Python version).
- **Vector Store (Chroma)**: Chunks + embeddings are stored in ChromaDB using the newer `PersistentClient` API (persistent on disk at `CHROMA_PERSIST_DIR`, defaults to `./chroma_db`). If you have older Chroma data you can migrate it with `chroma-migrate`.
- **Retrieval**: For each question the bot embeds the query, queries Chroma for top-k passages, and uses a distance threshold to decide whether the KB contains relevant information.
- **Answering**: The bot prefers OpenRouter (`OPENROUTER_API_KEY`) then OpenAI (`OPENAI_API_KEY`) and finally a local Ollama HTTP server (`USE_OLLAMA=true`) to generate concise answers constrained to the retrieved passages. If no LLM is configured the bot returns the retrieved passages directly.

**Environment / Run**

- **Required env**: `TELEGRAM_TOKEN` (bot token). Optional: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `USE_OLLAMA`, `OLLAMA_MODEL`, `CHROMA_PERSIST_DIR`, `LOG_LEVEL`.
- Create a virtualenv and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

- Run the bot:

```powershell
.venv\Scripts\python.exe bot.py
```

**Troubleshooting**

- If you see errors building `tokenizers` or other HF wheels on your Python version, use Python 3.11 (this repo has been tested under 3.11). Recreate the venv with `py -3.11 -m venv .venv` and reinstall requirements.
- If Chroma raises a deprecation/config error, run `chroma-migrate` to migrate old data, or let the bot create a fresh persistent DB at `CHROMA_PERSIST_DIR`.
- To debug extraction issues, enable logging or inspect the saved uploaded file in the working directory and run `python -c "from pdf_utils import extract_text_from_file; print(extract_text_from_file('path'))"`.

If you'd like, I can add a short example of uploading and asking via the Telegram UI, or enable provenance formatting in answers (include source chunk IDs). 

**Deploy to Render**

- Recommended service type: create a Background Worker (recommended for polling bots) in Render. If you prefer webhooks you'll need to expose an HTTP endpoint and switch to webhook mode.

- Steps (Background Worker / simple polling):
	1. Push your repo to GitHub and connect it in the Render dashboard (New → Background Worker).
	2. Set the Build Command to:

		 `pip install --upgrade pip && pip install -r requirements.txt`

	3. Set the Start Command to:

		 `python bot.py`

	4. Add required Environment Variables in Render's settings: `TELEGRAM_TOKEN` (required). Optional: `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `USE_OLLAMA`, `OLLAMA_MODEL`, `CHROMA_PERSIST_DIR`, `LOG_LEVEL`.

	5. (Optional) Enable a Persistent Disk for the service and set `CHROMA_PERSIST_DIR` to the mounted path (default `./chroma_db`) so ChromaDB persists between deploys.

- Notes / alternatives:
	- If you want webhooks instead of polling, deploy as a Web Service and implement a small HTTP handler that calls `ApplicationBuilder().webhook()` (not included in this repo). Use `gunicorn` or `uvicorn` as the start command for web services.
	- Monitor logs in Render to confirm indexing and to see messages like `Indexed X chunks from <filename>` and retrieval debug logs (when `LOG_LEVEL=DEBUG`).
	- If you see Chroma sqlite schema/lock errors on startup, stop the service, move or delete `chroma_db/chroma.sqlite3` (or enable Persistent Disk) and redeploy so the service can recreate the DB.

These steps will get the bot running on Render using the repository's current polling implementation. For production, consider running the vector DB and LLM behind managed services or enabling backups for the persistent disk.
