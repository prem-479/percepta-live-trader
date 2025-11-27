import os
import time
import jwt  # pip install pyjwt
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("GROWW_API_KEY")
API_SECRET = os.getenv("GROWW_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing API Key or Secret inside .env")

payload = {
    "apikey": API_KEY,
    "iat": int(time.time()),
    "exp": int(time.time()) + 24*60*60  # 1 day token
}

token = jwt.encode(payload, API_SECRET, algorithm="HS256")

print("\n🔐 GENERATED ACCESS TOKEN:\n")
print(token)

# Append token to .env
with open(".env", "a", encoding="utf-8") as f:
    f.write(f"\nGROWW_ACCESS_TOKEN={token}\n")

print("\n✔ Saved GROWW_ACCESS_TOKEN into .env")
