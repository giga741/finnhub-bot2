import os
import requests

BASE_URL = "https://api-capital.backend-capital.com"
API_KEY = os.getenv("CAPITAL_API_KEY")
IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER")  # la tua email di login
PASSWORD = os.getenv("CAPITAL_PASSWORD")      # la password API

headers = {
    "X-CAP-API-KEY": API_KEY,
    "Content-Type": "application/json"
}

payload = {
    "identifier": IDENTIFIER,
    "password": PASSWORD
}

print("👉 Provo login con:")
print("API_KEY:", API_KEY[:4] + "..." if API_KEY else None)
print("IDENTIFIER:", IDENTIFIER)
print("PASSWORD:", "****" if PASSWORD else None)

r = requests.post(f"{BASE_URL}/api/v1/session", headers=headers, json=payload)

print("Status code:", r.status_code)
print("Risposta:", r.text)
