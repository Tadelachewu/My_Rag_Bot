import os
import sys

MODE = os.getenv("MODE", "prod").strip().lower()

if MODE == "dev":
    # Run the bot in development polling mode (Telegram polling)
    try:
        import bot

        if hasattr(bot, "main"):
            bot.main()
        else:
            print("No bot.main() entrypoint found; cannot start in dev mode")
            sys.exit(1)
    except Exception as exc:
        print("Error starting bot in dev mode:", exc)
        raise

else:
    # Production: run FastAPI app with uvicorn. Render will supply PORT env var.
    port = int(os.getenv("PORT", "10000"))
    host = "0.0.0.0"
    # Import here to avoid adding extra deps when only running dev mode
    try:
        import uvicorn

        uvicorn.run("app:app", host=host, port=port, log_level="info")
    except Exception as exc:
        print("Failed to start uvicorn:", exc)
        raise
