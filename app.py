import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import List

from bot import (
    embed_texts,
    collection,
    WORKDIR,
    chunk_text,
    generate_answer_from_passages,
    build_telegram_application,
    start_telegram_polling,
    stop_telegram_polling,
    logger as bot_logger,
)
from pdf_utils import extract_text_from_file

_telegram_application = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _telegram_application
    enable = os.getenv("ENABLE_TELEGRAM_BOT", "true").strip().lower() in ("1", "true", "yes", "on")
    token_present = bool((os.getenv("TELEGRAM_TOKEN") or "").strip())
    if enable and token_present:
        try:
            _telegram_application = build_telegram_application()
            await start_telegram_polling(_telegram_application)
            bot_logger.info("Telegram polling started")
        except Exception:
            bot_logger.exception("Failed to start Telegram polling in web service mode")
            _telegram_application = None
    yield
    if _telegram_application is not None:
        try:
            await stop_telegram_polling(_telegram_application)
            bot_logger.info("Telegram polling stopped")
        except Exception:
            bot_logger.exception("Failed to stop Telegram polling cleanly")
        finally:
            _telegram_application = None


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
