import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Request
from fastapi.responses import JSONResponse
from typing import List
from telegram import Update

from bot import (
    embed_texts,
    get_collection,
    WORKDIR,
    chunk_text,
    generate_answer_from_passages,
    build_telegram_application,
    start_telegram_polling,
    stop_telegram_polling,
    start_telegram_webhook,
    stop_telegram_webhook,
    clear_all_data,
    logger as bot_logger,
)
from pdf_utils import extract_text_from_file

_telegram_application = None
_telegram_mode = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _telegram_application
    global _telegram_mode

    token_present = bool((os.getenv("TELEGRAM_TOKEN") or "").strip())
    if token_present:
        render_external_url = (os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")
        webhook_url = (os.getenv("TELEGRAM_WEBHOOK_URL") or "").strip()
        if not webhook_url and render_external_url:
            webhook_url = f"{render_external_url}/telegram/webhook"

        mode = (os.getenv("TELEGRAM_MODE") or "").strip().lower()
        if not mode:
            mode = "webhook" if webhook_url else ("polling" if os.getenv("MODE", "prod").strip().lower() == "dev" else "off")

        _telegram_mode = mode
        if mode == "polling":
            try:
                _telegram_application = build_telegram_application()
                await start_telegram_polling(_telegram_application)
                bot_logger.info("Telegram polling started")
            except Exception:
                bot_logger.exception("Failed to start Telegram polling")
                _telegram_application = None
        elif mode == "webhook":
            if not webhook_url:
                bot_logger.warning("TELEGRAM_MODE=webhook but no TELEGRAM_WEBHOOK_URL/RENDER_EXTERNAL_URL found; bot disabled")
            else:
                try:
                    _telegram_application = build_telegram_application()
                    secret = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip() or None
                    await start_telegram_webhook(_telegram_application, webhook_url=webhook_url, secret_token=secret)
                    bot_logger.info("Telegram webhook configured: %s", webhook_url)
                except Exception:
                    bot_logger.exception("Failed to start Telegram webhook")
                    _telegram_application = None
        else:
            _telegram_application = None
            bot_logger.info("Telegram bot disabled (TELEGRAM_MODE=%s)", mode)
    yield
    if _telegram_application is not None:
        try:
            if _telegram_mode == "webhook":
                await stop_telegram_webhook(_telegram_application)
                bot_logger.info("Telegram webhook stopped")
            else:
                await stop_telegram_polling(_telegram_application)
                bot_logger.info("Telegram polling stopped")
        except Exception:
            bot_logger.exception("Failed to stop Telegram polling cleanly")
        finally:
            _telegram_application = None
            _telegram_mode = None


app = FastAPI(title="MyBot Web API", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "mode": os.getenv("MODE", "prod")}

@app.head("/")
async def root_head():
    return Response(status_code=200)


@app.get("/health")
async def health():
    return {"status": "ok", "mode": os.getenv("MODE", "prod")}


@app.post("/ask")
async def ask(question: dict):
    collection = get_collection()
    q = question.get("question") if isinstance(question, dict) else None
    if not q:
        raise HTTPException(status_code=400, detail="Missing 'question' in request body")

    try:
        if collection.count() == 0:
            return JSONResponse(status_code=404, content={"error": "No documents uploaded"})
    except Exception:
        # if collection doesn't support count reliably
        pass

    try:
        q_emb = embed_texts([q])[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    results = collection.query(query_embeddings=[q_emb], n_results=5, include=["documents", "metadatas", "distances"])
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return JSONResponse(status_code=404, content={"answer": "Not found in uploaded documents."})

    passages = [f"Source: {m.get('source')}\n{d}" for m, d in zip(metas, docs)]
    answer = generate_answer_from_passages(q, passages)
    if answer is None:
        answer = "\n\n---\n\n".join(docs)

    sources = []
    for m in metas:
        s = m.get("source") if isinstance(m, dict) else None
        if s and s not in sources:
            sources.append(s)

    return {"answer": answer, "sources": sources}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    collection = get_collection()
    fname = file.filename or "uploaded_file"
    suffix = os.path.splitext(fname)[1].lower()
    if suffix not in (".pdf", ".pptx", ".docx"):
        raise HTTPException(status_code=400, detail="Only PDF, PPTX, or DOCX are supported")

    dest = WORKDIR / fname
    try:
        with open(dest, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    try:
        text = extract_text_from_file(str(dest))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {exc}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file")

    chunks = chunk_text(text)
    try:
        embeddings = embed_texts(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    ids = [f"{fname}--{i}" for i in range(len(chunks))]
    metadatas = [{"source": fname, "chunk": i} for i in range(len(chunks))]

    try:
        collection.delete(where={"source": fname})
    except Exception:
        pass

    collection.add(ids=ids, metadatas=metadatas, documents=chunks, embeddings=embeddings)
    try:
        persist_fn = getattr(collection._client if hasattr(collection, '_client') else None, "persist", None)
        # best-effort persist: some chroma clients expose persist on client not collection
        if callable(persist_fn):
            persist_fn()
    except Exception:
        pass

    return {"status": "ok", "indexed_chunks": len(chunks)}


@app.post("/clear")
async def clear():
    return clear_all_data()


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    if _telegram_application is None:
        raise HTTPException(status_code=503, detail="Telegram bot is not running")

    if _telegram_mode != "webhook":
        raise HTTPException(status_code=409, detail="Telegram bot is not in webhook mode")

    secret = (os.getenv("TELEGRAM_WEBHOOK_SECRET") or "").strip()
    if secret:
        got = (request.headers.get("X-Telegram-Bot-Api-Secret-Token") or "").strip()
        if got != secret:
            raise HTTPException(status_code=401, detail="Invalid secret token")

    payload = await request.json()
    update = Update.de_json(payload, _telegram_application.bot)
    await _telegram_application.process_update(update)
    return {"ok": True}
