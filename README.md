# x10 - Multi-Model AI Arena & Debate Chamber

[![Live Demo](https://img.shields.io/badge/Live%20Demo-x10ai.vercel.app-13925e?style=for-the-badge)](https://x10ai.vercel.app)

**x10** is an interactive, real-time AI multi-model interface that lets you broadcast prompts to 10 AI models in parallel, chat with individual models, and watch them engage in live debates inside the **Debate Chamber**. 

It is designed with a premium, responsive user interface optimized for both desktop grid views and mobile horizontal swipable chats.

---

## 🚀 Key Features

### 1. Broadcaster / Parallel Arena
* **Desktop Grid**: Send a single prompt and watch all 10 models generate streaming responses side-by-side in real-time.
* **Mobile Swipe View**: Swipe horizontally left-to-right to compare model responses in parallel on mobile viewports.

### 2. Focused Single Chat & Vision Input
* **Single Model Focus**: Isolate any of the 10 models for a dedicated chat experience.
* **Vision / Image Uploads**: Select models like **Gemini 2.0** or **Llama 4 Scout** to upload images (PNG/JPEG) and ask visual prompts.

### 3. AI Debate Chamber (The Council)
* **Live AI Debate**: Convene a turn-based live debate among the models on any topic.
* **Scribe's Ledger**: A dedicated "Scribe" compiles and updates a live summary of key arguments in real-time.
* **User Intervention**: Inject your own statements into the live debate at any turn.
* **Final Verdict**: The debate automatically concludes after 20 turns (or manually) and streams a structured consensus verdict.

---

## 🤖 Supported Models & Providers

| Model Name | API Provider / Backend | Vision Support |
| :--- | :--- | :---: |
| **Llama 3.3** | Groq API Key 1 (`llama-3.3-70b-versatile`) | ❌ |
| **Qwen 3.8** | Groq API Key 1 (`qwen/qwen3.8-27b`) | ❌ |
| **Orpheus Saudi** | Groq API Key 1 (`canopylabs/orpheus-arabic-saudi`) | ❌ |
| **GPT-OSS 20B** | Groq API Key 2 (`openai/gpt-oss-20b`) | ❌ |
| **Compound** | Groq API Key 2 (`groq/compound`) | ❌ |
| **GPT-OSS 120B** | Groq API Key 2 (`openai/gpt-oss-120b`) | ❌ |
| **Llama 4 Scout** | Groq API Key 2 (`meta-llama/llama-4-scout-17b-16e-instruct`) | ✅ |
| **Gemini 2.0** | Google Gemini API | ✅ |
| **GPT-4o** | Pollinations API (Stateless) | ❌ |
| **Llama 3.2** | Cloudflare Workers AI | ❌ |

---

## 🛠️ Setup & Local Installation

### 1. Clone & Install Dependencies
Ensure you have Python 3.8+ installed. Navigate to the project root and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a file named `.env` in the root folder and add your API credentials:
```env
GROQ_API_KEY_1=your_groq_key_1
GROQ_API_KEY_2=your_groq_key_2
GOOGLE_API_KEY=your_gemini_key
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
```

### 3. Run the App
Launch the FastAPI development server:
```bash
uvicorn main:app --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 🌐 Deployment to Vercel

The repository is configured for deployment on the Vercel platform using `@vercel/python`:

1. Import the repository into your Vercel Dashboard.
2. Configure your Environment Variables in the project settings.
3. Click **Deploy**. Vercel will automatically read `vercel.json` and configure serverless routing.
