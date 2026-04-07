from playwright.sync_api import sync_playwright
import time
import os

SESSION_PATH = "scripts/1_collecte/auth/facebook_session.json"

with sync_playwright() as p:
    navigateur = p.chromium.launch(headless=False)
    contexte = navigateur.new_context()
    page = contexte.new_page()

    page.goto("https://www.facebook.com/", timeout=60000)
    print("🔐 Connecte-toi manuellement à Facebook...")
    print("⏳ Tu as 60 secondes")

    time.sleep(120)  

    os.makedirs(os.path.dirname(SESSION_PATH), exist_ok=True)
    contexte.storage_state(path=SESSION_PATH)

    print("✅ Session Facebook sauvegardée")
    navigateur.close()
