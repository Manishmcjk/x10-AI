import os
import json
import requests
import aiohttp
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. CLIENT CONFIGURATION & INITIALIZATION ---

# Groq Client 1
GROQ_CLIENT_1 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_1") or "missing_key",
    timeout=5.0,
    max_retries=0
)

# Groq Client 2
GROQ_CLIENT_2 = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY_2") or "missing_key",
    timeout=5.0,
    max_retries=0
)

# Google Gemini Client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or "missing_key")

# Cloudflare Configuration
CF_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")
CF_HEADERS = {"Authorization": f"Bearer {CF_API_TOKEN}"}

# --- HELPER FUNCTIONS ---

def build_memory_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Constructs a conversation script from the message history.
    """
    prompt = ""
    for m in messages:
        role = "User" if m.get('role') == 'user' else "Assistant"
        content = m.get('content', '')
        if isinstance(content, list):
            text_parts = [p.get('text', '') for p in content if isinstance(p, dict) and 'text' in p]
            content = " ".join(text_parts)
        prompt += f"{role}: {content}\n"
    prompt += "Assistant:"
    return prompt

async def stream_cf_model(messages: List[Dict[str, str]], cf_model: str):
    """Stream from Cloudflare Workers AI with fallback"""
    url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/run/{cf_model}"
    full_text = build_memory_prompt(messages)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=CF_HEADERS, 
                json={"prompt": full_text, "stream": True, "max_tokens": 1024}
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        line_text = line.decode("utf-8").strip()
                        if line_text.startswith("data: "):
                            try:
                                data = json.loads(line_text[6:])
                                if "response" in data and data["response"] is not None:
                                    yield str(data["response"])
                            except:
                                pass
                    return
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except Exception:
        pass

# --- 2. GENERATOR FUNCTIONS (STREAMING LOGIC) ---

async def clean_think_filter(generator):
    """Strip out any <think>...</think> blocks from the streaming response"""
    in_think = False
    
    async for chunk in generator:
        if in_think:
            if "</think>" in chunk:
                in_think = False
                parts = chunk.split("</think>", 1)
                if len(parts) > 1 and parts[1]:
                    yield parts[1]
            continue
            
        if "<think>" in chunk:
            parts = chunk.split("<think>", 1)
            if parts[0]:
                yield parts[0]
            in_think = True
            if "</think>" in parts[1]:
                in_think = False
                after_parts = parts[1].split("</think>", 1)
                if len(after_parts) > 1 and after_parts[1]:
                    yield after_parts[1]
        elif "</think>" in chunk:
            parts = chunk.split("</think>", 1)
            if len(parts) > 1 and parts[1]:
                yield parts[1]
        else:
            yield chunk

# === GROQ ENGINE 1 MODELS (WITH CLOUDFLARE FALLBACK) ===

async def generate_groq_llama(messages):
    """Stream Llama 3.3 70B (Groq -> Cloudflare Fallback)"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"):
            yield chunk

async def generate_groq_qwen38(messages):
    """Stream Qwen 3.8 / DeepSeek R1 (Groq -> Cloudflare Fallback)"""
    async def raw_generator():
        try:
            stream = await GROQ_CLIENT_1.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            async for chunk in stream_cf_model(messages, "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"):
                yield chunk
            
    async for chunk in clean_think_filter(raw_generator()):
        yield chunk

async def generate_groq_orpheus(messages):
    """Stream Orpheus Saudi / Llama 3.1 8B (Groq -> Cloudflare Fallback)"""
    try:
        stream = await GROQ_CLIENT_1.chat.completions.create(
            model="canopylabs/orpheus-arabic-saudi",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        try:
            stream = await GROQ_CLIENT_1.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.1-8b-instruct"):
                yield chunk

# === GROQ ENGINE 2 MODELS (WITH CLOUDFLARE FALLBACK) ===

async def generate_groq_gptoss20(messages):
    """Stream OpenAI GPT-OSS 20B / Mistral 7B (Groq -> Cloudflare Fallback)"""
    async def raw_generator():
        try:
            stream = await GROQ_CLIENT_2.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception:
            async for chunk in stream_cf_model(messages, "@cf/mistral/mistral-7b-instruct-v0.1"):
                yield chunk
            
    async for chunk in clean_think_filter(raw_generator()):
        yield chunk

async def generate_groq_compound(messages):
    """Stream Groq Compound / Llama 3.1 70B (Groq -> Cloudflare Fallback)"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="groq/compound",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.1-70b-instruct"):
            yield chunk

async def generate_groq_gptoss(messages):
    """Stream OpenAI GPT-OSS 120B / Llama 3.3 70B (Groq -> Cloudflare Fallback)"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"):
            yield chunk

# === OTHER PROVIDERS ===

async def generate_groq_scout(messages):
    """Stream Llama 4 Scout 17B / Llama 3.2 3B (Groq -> Cloudflare Fallback)"""
    try:
        stream = await GROQ_CLIENT_2.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            stream=True
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception:
        async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.2-3b-instruct"):
            yield chunk

async def generate_gemini(messages, image=None):
    """Stream Gemini (Auto-switching versions with Cloudflare & Pollinations fallback)"""
    gemini_history = []
    for m in messages[:-1]:
        role = "user" if m.get("role") == "user" else "model"
        gemini_history.append({"role": role, "parts": [m.get("content", "")]})
    
    current_message = messages[-1].get("content", "") if messages else ""

    import base64
    image_part = None
    if image and "data" in image and "mime_type" in image:
        try:
            image_data = base64.b64decode(image["data"])
            image_part = {
                "mime_type": image["mime_type"],
                "data": image_data
            }
        except Exception:
            pass

    # Priority list of Gemini models
    models_to_try = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    for model_id in models_to_try:
        try:
            model = genai.GenerativeModel(model_id)
            if image_part:
                contents = []
                for h in gemini_history:
                    contents.append({"role": h["role"], "parts": h["parts"]})
                contents.append({"role": "user", "parts": [current_message, image_part]})
                
                response = await model.generate_content_async(contents, stream=True)
            else:
                chat = model.start_chat(history=gemini_history)
                response = await chat.send_message_async(current_message, stream=True)
                
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
            return
        except Exception:
            continue
            
    # Fallback 1: Cloudflare Workers AI (Llama 3.2)
    try:
        has_yielded = False
        async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.2-3b-instruct"):
            has_yielded = True
            yield chunk
        if has_yielded:
            return
    except Exception:
        pass

    # Fallback 2: Pollinations AI
    async for chunk in generate_pollinations(messages):
        yield chunk

async def generate_pollinations(messages):
    """Stream Pollinations (GPT-4o Proxy) - Stateless"""
    full_text = build_memory_prompt(messages)
    encoded_prompt = requests.utils.quote(full_text)
    url = f"https://text.pollinations.ai/{encoded_prompt}?model=openai"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            yield chunk.decode('utf-8')
                    return
    except (GeneratorExit, asyncio.CancelledError):
        raise
    except Exception:
        pass

    async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"):
        yield chunk

async def generate_cloudflare(messages):
    """Stream Cloudflare Workers AI (Llama 3.2) - Stateless"""
    async for chunk in stream_cf_model(messages, "@cf/meta/llama-3.2-3b-instruct"):
        yield chunk

# --- 3. ROUTE HANDLERS ---

@app.get("/health")
async def health_check():
    """Endpoint for system health checks"""
    return {"status": "ok"}

@app.get("/search")
async def dummy_search(q: str = ""):
    """Dummy endpoint for system search checks"""
    return {"results": []}

@app.get("/")
async def get_ui():
    """Serves the main HTML interface"""
    with open("index.html", "r", encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/meeting.png")
async def get_meeting_logo():
    """Serves the meeting logo image file"""
    return FileResponse("meeting.png")

@app.get("/favicon.ico")
async def get_favicon():
    """Serves the meeting logo as favicon icon"""
    return FileResponse("meeting.png", media_type="image/png")

@app.post("/chat/{provider}")
async def chat_endpoint(provider: str, request: Request):
    """Main routing endpoint for all AI models"""
    try:
        data = await request.json()
        messages = data.get("messages", [])
        image = data.get("image", None)

        # Groq Key 1 Routes
        if provider == "groq_llama":
            return StreamingResponse(generate_groq_llama(messages), media_type="text/plain")
        if provider in ["groq_qwen38", "groq_qwen36"]:
            return StreamingResponse(generate_groq_qwen38(messages), media_type="text/plain")
        if provider in ["groq_orpheus", "groq_allam"]:
            return StreamingResponse(generate_groq_orpheus(messages), media_type="text/plain")
        
        # Groq Key 2 Routes
        if provider in ["groq_gptoss20", "groq_qwen"]:
            return StreamingResponse(generate_groq_gptoss20(messages), media_type="text/plain")
        if provider == "groq_compound":
            return StreamingResponse(generate_groq_compound(messages), media_type="text/plain")
        if provider == "groq_gptoss":
            return StreamingResponse(generate_groq_gptoss(messages), media_type="text/plain")
        
        # Other Provider Routes
        if provider == "groq_scout":
            if image and "data" in image and "mime_type" in image:
                messages[-1]["content"] = [
                    {"type": "text", "text": messages[-1]["content"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image['mime_type']};base64,{image['data']}"
                        }
                    }
                ]
            return StreamingResponse(generate_groq_scout(messages), media_type="text/plain")
        if provider == "gemini":
            return StreamingResponse(generate_gemini(messages, image), media_type="text/plain")
        if provider == "poll":
            return StreamingResponse(generate_pollinations(messages), media_type="text/plain")
        if provider == "cf":
            return StreamingResponse(generate_cloudflare(messages), media_type="text/plain")
        
        return "Invalid Provider ID"
        
    except Exception as e:
        return f"Server Error: {str(e)}"

# --- 4. DEBATE CHAMBER ENDPOINTS ---

MODEL_GENERATORS = {
    "groq_llama": (generate_groq_llama, "Llama 3.3"),
    "groq_qwen38": (generate_groq_qwen38, "Qwen 3.8"),
    "groq_qwen36": (generate_groq_qwen38, "Qwen 3.8"),
    "groq_orpheus": (generate_groq_orpheus, "Orpheus Saudi"),
    "groq_allam": (generate_groq_orpheus, "Orpheus Saudi"),
    "groq_gptoss20": (generate_groq_gptoss20, "GPT-OSS 20B"),
    "groq_qwen": (generate_groq_gptoss20, "GPT-OSS 20B"),
    "groq_compound": (generate_groq_compound, "Compound"),
    "groq_gptoss": (generate_groq_gptoss, "GPT-OSS 120B"),
    "groq_scout": (generate_groq_scout, "Llama 4 Scout"),
    "gemini": (generate_gemini, "Gemini 2.0"),
    "poll": (generate_pollinations, "GPT-4o"),
    "cf": (generate_cloudflare, "Llama 3.2"),
}

def build_debate_messages(topic: str, history: List[Dict[str, Any]], current_model_id: str, current_model_name: str, turns_left: int) -> List[Dict[str, str]]:
    system_instruction = (
        f"You are {current_model_name}, a critical and open-minded member of the AI Council.\n"
        f"You are debating the topic: \"{topic}\".\n"
        "Your goal: Select a clear stance on this topic and persistently defend it, build on your previous logic, and directly attack opponent weaknesses.\n"
        f"Debate Progress: There are exactly {turns_left} messages remaining in this entire debate (all models combined) before the Scribe calls a final verdict. budget your arguments to prove your point before the debate ends.\n"
        f"{'This is the absolute FINAL speaker turn. Make your final strike at the opposing arguments and conclude your case.' if turns_left == 1 else ''}\n\n"
        "Rhetorical Directives (MANDATORY FORMAT):\n"
        "1. You MUST follow this exact structure (including double line breaks and bullets):\n"
        "**STANCE**: [Your stance or opinion in 3-5 words]\n\n"
        "**REASONING**:\n"
        "* [First key premise or direct refutation targeting opponents' logic]\n"
        "* [Supporting fact or logical deduction]\n\n"
        "2. Strict Word Limit: Stay under 45 words total. Be extremely concise. Avoid all preambles, greetings, intros, or filler text. Start directly with '**STANCE**'.\n"
        "Ideological Directives:\n"
        "3. Reject generic 'woke' platitudes, sugarcoating, or modern sanitized corporate speech. Speak with core realistic, scientific, factual, and critical philosophical rigor.\n"
        "4. Base your arguments on facts, logic, and critical evaluation, even if the conclusion is controversial or uncomfortable.\n"
        "5. Speak in the first person as yourself."
    )
    
    formatted_messages = []
    formatted_messages.append({"role": "system", "content": system_instruction})
    
    if not history:
        # First turn
        formatted_messages.append({
            "role": "user", 
            "content": f"The debate topic is: \"{topic}\". Please present your opening statement."
        })
    else:
        # Build chronological transcript
        transcript = f"Here is the debate transcript so far on the topic \"{topic}\":\n\n"
        for turn in history:
            speaker_name = turn.get("name", "Unknown AI")
            content = turn.get("content", "")
            transcript += f"--- Speaker: {speaker_name} ---\n{content}\n\n"
            
        transcript += f"It is now your turn, {current_model_name}. Present your counter-argument or supporting points."
        formatted_messages.append({"role": "user", "content": transcript})
        
    return formatted_messages

async def clean_stream_filter(generator):
    """Discard all output before actual STANCE marker to strip thinking/preambles"""
    buffer = ""
    yielded_stance = False
    
    async for chunk in generator:
        if yielded_stance:
            yield chunk
            continue
            
        buffer += chunk
        lower_buf = buffer.lower()
        
        # Look for explicit stance markers (with colons or stars) to avoid false matches in thinking texts
        idx = -1
        for marker in ["**stance**:", "stance:", "**stance**", "stance :"]:
            if marker in lower_buf:
                idx = lower_buf.find(marker)
                break
                
        if idx != -1:
            stars_idx = buffer.rfind("**", 0, idx)
            start_pos = stars_idx if (stars_idx != -1 and idx - stars_idx <= 3) else idx
            yielded_stance = True
            yield buffer[start_pos:]
            buffer = ""
        else:
            if len(buffer) > 1200:
                yielded_stance = True
                yield buffer
                buffer = ""

@app.post("/debate/next")
async def debate_next_endpoint(request: Request):
    """Generates the next turn in the AI Council debate"""
    try:
        data = await request.json()
        topic = data.get("topic", "")
        history = data.get("history", [])
        model_id = data.get("model_id", "")
        
        # Calculate turns left (limit to 20 messages total)
        turns_left = max(1, 20 - len(history))
        
        if model_id not in MODEL_GENERATORS:
            return "Invalid Model ID"
            
        generator_func, model_name = MODEL_GENERATORS[model_id]
        messages = build_debate_messages(topic, history, model_id, model_name, turns_left)
        
        return StreamingResponse(
            clean_stream_filter(generator_func(messages)), 
            media_type="text/plain",
            headers={
                "X-Speaker-Id": model_id,
                "X-Speaker-Name": model_name
            }
        )
    except Exception as e:
        return f"Debate Turn Error: {str(e)}"

@app.post("/debate/verdict")
async def debate_verdict_endpoint(request: Request):
    """Summarizes and synthesizes the entire debate into a final verdict"""
    try:
        data = await request.json()
        topic = data.get("topic", "")
        history = data.get("history", [])
        
        # Build the final debate transcript
        transcript = f"Debate Topic: \"{topic}\"\n\n"
        for turn in history:
            speaker_name = turn.get("name", "Unknown AI")
            content = turn.get("content", "")
            transcript += f"[{speaker_name}]: {content}\n\n"
            
        system_instruction = (
            "You are the Scribe of the AI Council. Your job is to listen to the debate transcript "
            "and write a structured, beautifully formatted final verdict.\n"
            "Format your output in Markdown with the following sections:\n"
            "1. **Core Conflict**: A summary of the central debate tension.\n"
            "2. **Key Arguments Pro**: Highlight the main supporting points.\n"
            "3. **Key Arguments Con**: Highlight the main counter-points.\n"
            "4. **Areas of Consensus**: What did the models agree on?\n"
            "5. **Unresolved Disputes**: Where was the disagreement most intense?\n"
            "6. **Final Resolution**: Synthesize the debate into a final compromise or balanced verdict."
        )
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Here is the transcript of the debate:\n\n{transcript}\n\nScribe, please compile your final verdict."}
        ]
        
        return StreamingResponse(generate_gemini(messages), media_type="text/plain")
        
    except Exception as e:
        return f"Verdict Error: {str(e)}"