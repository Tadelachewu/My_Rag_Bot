import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, Document, BotCommand, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import chromadb
# Use the helper constructors exported by chromadb (PersistentClient/EphemeralClient)
import requests
from typing import Optional
import openai
import json
import re
load_dotenv()

# Basic logging setup must be available before other initialization
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# reduce noisy libraries
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.INFO)

# Embedding/OpenAI defaults used by embed_texts
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embed_model = None

import sqlite3

# Environment
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
from pathlib import Path as _Path
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").strip().lower() in ("1", "true", "yes")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
WORKDIR = Path(os.getenv("WORKDIR", "Saved"))
WORKDIR.mkdir(parents=True, exist_ok=True)
import json
import sys
import traceback
from pdf_utils import extract_text_from_file, chunk_text
def run_openrouter_embeddings(texts, model: Optional[str] = None):
    """Call OpenRouter embeddings endpoint and return list of embeddings.

    Expects `OPENROUTER_API_KEY` to be set in environment. Uses a default
    embedding model name when not provided.
    """
    key = OPENROUTER_API_KEY
    if not key:
        return None
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
# Build and send request to OpenRouter embeddings endpoint
    payload = {"model": model or "text-embedding-3-small", "input": texts}
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60.0)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter embeddings error {resp.status_code}: {resp.text}")
    data = resp.json()
    # Expecting data['data'] -> list of {'embedding': [...]} 
    return [item.get("embedding") for item in data.get("data", [])]
# ChromaDB persistent client / collection setup
CHROMA_PERSIST_DIR = os.getenv('CHROMA_PERSIST_DIR', 'chroma_db')
persist_dir = _Path(CHROMA_PERSIST_DIR)
persist_dir.mkdir(parents=True, exist_ok=True)

collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'documents')
try:
    PersistentClient = getattr(chromadb, "PersistentClient", None)
    if PersistentClient is not None:
        client = PersistentClient(path=str(persist_dir))
    else:
        # fallback to generic client (may be ephemeral)
        client = chromadb.Client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.create_collection(name=collection_name)
except Exception as exc:
    msg = str(exc) or ""
    # Handle older Chroma sqlite schema mismatch (common when upgrading versions)
    if isinstance(exc, sqlite3.OperationalError) or "no such column: collections.topic" in msg:
        dbfile = persist_dir / "chroma.sqlite3"
        if dbfile.exists():
            bak = persist_dir / "chroma.sqlite3.bak"
            try:
                os.replace(dbfile, bak)
                logger.warning("Backed up old ChromaDB sqlite to %s due to schema error", str(bak))
            except Exception:
                logger.exception("Failed to back up ChromaDB sqlite file")
        # retry creating a fresh persistent client and collection
        try:
            if PersistentClient is not None:
                client = PersistentClient(path=str(persist_dir))
            else:
                client = chromadb.Client()
            collection = client.create_collection(name=collection_name)
        except Exception:
            logger.exception("Failed to recreate ChromaDB client after backing up sqlite")
            client = chromadb.Client()
            try:
                collection = client.get_collection(name=collection_name)
            except Exception:
                collection = client.create_collection(name=collection_name)
    else:
        logger.exception("Failed to initialize ChromaDB client; continuing without persistence")
        client = chromadb.Client()
        try:
            collection = client.get_collection(name=collection_name)
        except Exception:
            collection = client.create_collection(name=collection_name)

def embed_texts(texts):
    """Return list of embeddings for `texts`.

    Preference order:
      1. OpenRouter (if `OPENROUTER_API_KEY` set)
      2. OpenAI (if `OPENAI_API_KEY` set) -- kept as optional fallback
      3. Local `sentence-transformers` fallback (if installed)
    """
    # 1) OpenRouter
    if OPENROUTER_API_KEY:
        try:
            embs = run_openrouter_embeddings(texts)
            if embs:
                return embs
        except Exception:
            logger.exception("OpenRouter embeddings failed, falling back")

    # 2) OpenAI (optional fallback if user still has key)
    if OPENAI_API_KEY:
        try:
            # lazy import of openai to avoid dependency when not used
            import openai as _openai
            _openai.api_key = OPENAI_API_KEY
            resp = _openai.Embedding.create(model="text-embedding-3-small", input=texts)
            return [d["embedding"] for d in resp["data"]]
        except Exception:
            logger.exception("OpenAI embeddings failed, falling back")

    # 3) Local sentence-transformers fallback
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        raise RuntimeError(
            "No remote embedding key available and `sentence-transformers` is not installed. "
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY, or install sentence-transformers on a compatible Python version."
        )

    global embed_model
    if embed_model is None:
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded local embedding model: all-MiniLM-L6-v2")
    return embed_model.encode(texts).tolist()



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = ReplyKeyboardMarkup([[KeyboardButton("Upload PDF") , KeyboardButton("Ask Question")]], resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text(
        "Welcome — send PDF files (as Documents) to upload/update the knowledge base, then ask questions.\n\n"
        "Use the buttons below or the paperclip icon to upload PDFs.",
        reply_markup=keyboard,
    )

async def upload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Please send a PDF file as a Document (use the paperclip/attachment icon).")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send your question as a plain text message; I'll answer based only on uploaded PDFs.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Upload PDFs and then ask questions. Replies are only from your documents.")

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    doc: Document = update.message.document
    fname = doc.file_name or "uploaded_file"
    suffix = Path(fname).suffix.lower()
    if suffix not in ('.pdf', '.pptx', '.docx'):
        await update.message.reply_text("Please upload a PDF, PPTX, or DOCX document.")
        return

    file = await context.bot.get_file(doc.file_id)
    local_path = WORKDIR / fname
    await file.download_to_drive(custom_path=str(local_path))
    await update.message.reply_text(f"Saved {doc.file_name}, processing...")

    try:
        text = extract_text_from_file(str(local_path))
    except Exception as exc:
        logger.exception("Failed to extract text from uploaded file: %s", exc)
        await update.message.reply_text("Failed to extract text from the uploaded file.")
        return
    if not text.strip():
        await update.message.reply_text("No extractable text found in this PDF.")
        return

    # semantic chunking: prefer sentence-aware chunks with overlap
    chunks = chunk_text(text)
    logger.info("Extracted text length=%d; creating %d chunks", len(text), len(chunks))
    embeddings = embed_texts(chunks)

    ids = [f"{doc.file_name}--{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc.file_name, "chunk": i} for i in range(len(chunks))]

    collection.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
    # Persist if the client exposes a persist method (some client implementations do)
    try:
        persist_fn = getattr(client, "persist", None)
        if callable(persist_fn):
            persist_fn()
    except Exception:
        logger.exception("Failed to persist ChromaDB to disk")

    await update.message.reply_text(f"Indexed {len(chunks)} chunks from {doc.file_name}.")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text.strip()
    if question.startswith('/'):
        return

    if collection.count() == 0:
        await update.message.reply_text("No documents uploaded yet.")
        return

    q_emb = embed_texts([question])[0]
    # retrieve top 5
    results = collection.query(query_embeddings=[q_emb], n_results=5, include=['documents','metadatas','distances'])
    docs = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]

    if not docs:
        await update.message.reply_text("Not found in uploaded documents.")
        return

    # distance threshold (smaller is closer for cosine in chroma).
    # If `CHROMA_DISTANCE_THRESHOLD` is set (>=0) we will enforce it; otherwise
    # we allow the top passages through to the LLM for final determination.
    thresh = os.getenv('CHROMA_DISTANCE_THRESHOLD')
    try:
        thresh_v = float(thresh) if thresh is not None else -1.0
    except Exception:
        thresh_v = -1.0
    if distances and thresh_v >= 0 and distances[0] > thresh_v:
        await update.message.reply_text("Not found in uploaded documents.")
        return

    # Debug: log retrieval distances and top metas/docs for diagnostics
    logger.debug("Query distances=%s", distances)
    try:
        metas = results.get('metadatas', [[]])[0]
    except Exception:
        metas = []
    logger.debug("Top metas=%s", metas[:3])
    logger.debug("Top docs excerpts=%s", [d[:300] for d in docs[:3]])

    # Prepare context passages for the LLM (include source in the context only)
    metas = results.get('metadatas', [[]])[0]
    passages = [f"Source: {m.get('source')}\n{d}" for m, d in zip(metas, docs)]
    # Build a deduplicated list of source document names for the user-facing reply
    sources = []
    for m in metas:
        s = m.get('source') if isinstance(m, dict) else None
        if s and s not in sources:
            sources.append(s)

    if USE_OLLAMA:
        prompt_context = "\n\n---\n\n".join(passages)
        # Keep prompt size reasonable
        if len(prompt_context) > 6000:
            prompt_context = prompt_context[-6000:]

        prompt = (
            "You are an assistant that must answer using ONLY the provided documents.\n"
            f"Context:\n{prompt_context}\n\nQuestion: {question}\n\nAnswer concisely, and if the answer is not contained in the context, reply 'Not found in uploaded documents.'"
        )

        llm_out = run_ollama(prompt, model=OLLAMA_MODEL)
        if llm_out is None:
            await update.message.reply_text("LLM error: Ollama not available or failed to run.")
            return

        # Truncate long llm outputs for telegram
        if len(llm_out) > 3500:
            llm_out = llm_out[:3500] + "\n\n...truncated"

        # Always append short source list (document names only) to the reply
        max_show = 5
        shown = sources[:max_show]
        more = len(sources) - len(shown)
        src_line = "Sources: " + (", ".join(shown) + (f" and {more} more" if more > 0 else "")) if shown else ""
        reply = llm_out
        if src_line:
            reply = reply + "\n\n" + src_line
        await update.message.reply_text(reply)
        return

    # Use an LLM (OpenRouter/OpenAI/Ollama) to answer using only the retrieved passages.
    answer = generate_answer_from_passages(question, passages)
    if answer is None:
        # fallback to raw passages (no LLM configured)
        answer = "\n\n---\n\n".join([d for d in docs])

    if len(answer) > 3500:
        answer = answer[:3500] + "\n\n...truncated"

    # Append deduplicated source document names (limit displayed list)
    max_show = 5
    shown = sources[:max_show]
    more = len(sources) - len(shown)
    src_line = "Sources: " + (", ".join(shown) + (f" and {more} more" if more > 0 else "")) if shown else ""
    reply = answer
    if src_line:
        reply = reply + "\n\n" + src_line
    await update.message.reply_text(reply)


def generate_answer_from_passages(question: str, passages: list[str]) -> Optional[str]:
    """Generate a concise answer using an available LLM and the provided passages.

    The LLM is instructed to answer using ONLY the provided passages. If no remote
    LLM is configured, returns None to indicate fallback to raw passages.
    """
    context = "\n\n---\n\n".join(passages)
    # keep context size bounded
    if len(context) > 6000:
        context = context[-6000:]

    system_prompt = (
        "You are an assistant that must answer using ONLY the provided documents. "
        "If the answer is not contained in the documents, reply exactly: 'Not found in uploaded documents.'"
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely."

    # Prefer OpenRouter if configured
    if OPENROUTER_API_KEY:
        out = run_openrouter(f"{system_prompt}\n\n{user_prompt}")
        if out is not None:
            return out

    # Use OpenAI ChatCompletion if API key available
    if OPENAI_API_KEY:
        try:
            chat_resp = openai.ChatCompletion.create(
                model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo'),
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            msg = chat_resp['choices'][0]['message']['content']
            return msg
        except Exception:
            logger.exception("OpenAI chat completion failed")

    # Fallback to Ollama if configured
    if USE_OLLAMA:
        out = run_ollama(f"{system_prompt}\n\n{user_prompt}")
        if out is not None:
            return out

    return None


def run_ollama(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Call the local Ollama HTTP API at http://localhost:11434/api/generate.

    Returns the generated text or None on error.
    """
    model = model or OLLAMA_MODEL
    url = os.getenv("OLLAMA_HTTP_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 512,
    }
    try:
        # Use streaming to handle incremental outputs from Ollama if available
        resp = requests.post(url, json=payload, timeout=120.0, stream=True)
        if resp.status_code != 200:
            logger.error("Ollama HTTP error %s: %s", resp.status_code, resp.text)
            return None

        # If server returned JSON in one shot, prefer that
        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            try:
                data = resp.json()
                if isinstance(data, dict):
                    return data.get("text") or data.get("output") or str(data)
                return str(data)
            except Exception:
                pass

        # Otherwise assemble streamed lines (could be SSE or newline-delimited JSON)
        out_parts: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            # try parsing JSON fragments
            try:
                j = json.loads(line)
                # Ollama streaming fragments may include `response` or `text` fields
                part = None
                if isinstance(j, dict):
                    part = j.get("response") or j.get("text") or j.get("output")
                if part:
                    out_parts.append(part)
                else:
                    # fallback to stringified JSON
                    out_parts.append(json.dumps(j))
            except Exception:
                # not JSON, append raw text
                out_parts.append(line)

        out = "".join(out_parts)
        # Remove any accidental JSON telemetry/metadata fragments that start with {"model":
        m = re.search(r'\{\s*"model"\s*:', out)
        if m:
            out = out[:m.start()].strip()
        return out
    except Exception as exc:
        logger.exception("Error calling Ollama HTTP API: %s", exc)
        return None


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free")

def run_openrouter(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Call OpenRouter chat completions and return the assistant text."""
    key = OPENROUTER_API_KEY
    if not key:
        return None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"enabled": True},
        "max_tokens": 512,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60.0)
        if resp.status_code != 200:
            logger.error("OpenRouter HTTP error %s: %s", resp.status_code, resp.text)
            return None
        data = resp.json()
        # Expecting structure similar to OpenAI: choices[0].message
        choice = data.get("choices", [None])[0]
        if not choice:
            return str(data)
        message = choice.get("message") or {}
        return message.get("content") or message.get("text") or str(message)
    except Exception as exc:
        logger.exception("Error calling OpenRouter API: %s", exc)
        return None


def choose_llm(prompt: str) -> Optional[str]:
    """Prefer OpenRouter if configured, else fall back to Ollama local HTTP API."""
    if OPENROUTER_API_KEY:
        out = run_openrouter(prompt)
        if out is not None:
            return out
    return run_ollama(prompt)

def main():
    async def on_startup(app):
        try:
            cmds = [
                BotCommand("start", "Start"),
                BotCommand("upload", "Upload a PDF"),
                BotCommand("ask", "Ask a question"),
                BotCommand("help", "Help"),
            ]
            await app.bot.set_my_commands(cmds)
            logger.info("Registered bot commands: %s", [c.command for c in cmds])
        except Exception:
            logger.exception("Failed to register bot commands")

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).post_init(on_startup).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("upload", upload_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Startup info
    masked = (TELEGRAM_TOKEN[:6] + "...") if TELEGRAM_TOKEN else "(none)"
    try:
        logger.info("Bot starting with token: %s", masked)
        logger.info("Chroma persist dir: %s", persist_dir)
        try:
            count = collection.count()
        except Exception:
            count = "unknown"
        logger.info("Chroma collection '%s' count: %s", collection_name, count)
        logger.info("USE_OLLAMA=%s, OLLAMA_MODEL=%s", USE_OLLAMA, OLLAMA_MODEL)
        logger.info("Starting polling (press Ctrl+C to stop)...")
        app.run_polling()
    except Exception:
        logger.exception("Bot crashed during run")
        raise

if __name__ == '__main__':
    main()
