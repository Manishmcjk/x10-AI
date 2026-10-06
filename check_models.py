import os
import sys
import requests
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

def test_groq_1():
    print("1. Testing Groq Key 1 (Llama 3.3 / Qwen 3.8)...", end=" ")
    key = os.getenv("GROQ_API_KEY_1")
    if not key:
        print("[FAIL] Missing GROQ_API_KEY_1")
        return
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2
        )
        print("[OK] Success (llama-3.3-70b-versatile)")
    except Exception as e:
        print(f"[FAIL] {e}")

def test_groq_2():
    print("2. Testing Groq Key 2 (GPT-OSS 120B / Compound)...", end=" ")
    key = os.getenv("GROQ_API_KEY_2")
    if not key:
        print("[FAIL] Missing GROQ_API_KEY_2")
        return
    try:
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
        client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2
        )
        print("[OK] Success (openai/gpt-oss-120b)")
    except Exception as e:
        print(f"[FAIL] {e}")

def test_gemini():
    print("3. Testing Gemini...", end=" ")
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        print("[FAIL] Missing GOOGLE_API_KEY")
        return
    try:
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        model.generate_content("hi")
        print("[OK] Success (gemini-2.0-flash)")
    except Exception as e:
        print(f"[FAIL] {e}")

def test_cloudflare():
    print("4. Testing Cloudflare Workers AI...", end=" ")
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not token:
        print("[FAIL] Missing Cloudflare credentials")
        return
    try:
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.2-3b-instruct"
        resp = requests.post(url, headers=headers, json={"prompt": "hi", "max_tokens": 5}, timeout=10)
        if resp.status_code == 200:
            print("[OK] Success (@cf/meta/llama-3.2-3b-instruct)")
        else:
            print(f"[FAIL] Status {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] {e}")

def test_pollinations():
    print("5. Testing Pollinations AI...", end=" ")
    try:
        resp = requests.get("https://text.pollinations.ai/hi?model=openai", timeout=10)
        if resp.ok:
            print("[OK] Success (GPT-4o via Pollinations)")
        else:
            print(f"[FAIL] Status {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] {e}")

if __name__ == "__main__":
    test_groq_1()
    test_groq_2()
    test_gemini()
    test_cloudflare()
    test_pollinations()