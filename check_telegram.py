import os
import sys
from dotenv import load_dotenv
import requests

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    print("TELEGRAM_TOKEN not set in environment or .env")
    sys.exit(1)

url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
try:
    resp = requests.get(url, timeout=10)
    print("HTTP", resp.status_code)
    print(resp.text)
except Exception as e:
    print("Error calling Telegram API:", e)
    sys.exit(2)
