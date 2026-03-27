#!/usr/bin/env python3
"""
MARCUS — Digital Co-Founder
Neuradex AI | Bo Bell
Brain: MiniMax M2.5 via NVIDIA NIM | Memory: mem0 + NVIDIA Embeddings | Interface: Telegram
"""

# ─── IMPORTS ───────────────────────────────────────────────────────────────────
import asyncio, base64, json, logging, os, imaplib
import email as elib, subprocess, time, urllib.parse
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Optional

import httpx, pytz
from bs4 import BeautifulSoup
from openai import OpenAI
from telegram import Update, Bot
from telegram.ext import (
    Application, MessageHandler, CommandHandler,
    filters, ContextTypes
)
from telegram.constants import ParseMode

# ─── CONFIG ────────────────────────────────────────────────────────────────────
NVIDIA_KEY     = os.getenv("NVIDIA_KEY",     "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
GITHUB_TOKEN   = os.getenv("GITHUB_TOKEN",   "")
VERCEL_TOKEN   = os.getenv("VERCEL_TOKEN",   "")
TAVILY_KEY     = os.getenv("TAVILY_KEY",     "")
TEXTBELT_KEY   = os.getenv("TEXTBELT_KEY",   "")
GOOGLE_KEY     = os.getenv("GOOGLE_KEY",     "")
RESEND_KEY     = os.getenv("RESEND_KEY",     "")
GMAIL_USER     = os.getenv("GMAIL_USER",     "bobellconstulting@gmail.com")
GMAIL_PASS     = os.getenv("GMAIL_PASS",     "")
BO_CHAT_ID     = int(os.getenv("BO_CHAT_ID", "7240677590"))

MODEL    = "minimaxai/minimax-m2.5"
NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
CT      = pytz.timezone("America/Chicago")

# ─── PATHS ─────────────────────────────────────────────────────────────────────
BASE          = Path("/tmp/neuradex/marcus")
SOUL_F        = BASE / "SOUL.md"
MEM_F         = BASE / "MEMORY.md"
MARKET_F      = BASE / "MARKET.md"
HIST_F        = BASE / "HISTORY.json"
CHATID_F      = BASE / "bo_chat_id.txt"
REMIND_F      = BASE / "reminders.json"
MARCUS_MEM_DIR = Path("/tmp/marcus_memory")
BASE.mkdir(parents=True, exist_ok=True)
MARCUS_MEM_DIR.mkdir(exist_ok=True)

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/tmp/marcus.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("marcus")

# ─── AI CLIENT ─────────────────────────────────────────────────────────────────
oai = OpenAI(
    api_key=NVIDIA_KEY,
    base_url=NVIDIA_BASE
)

# ─── SEMANTIC MEMORY (mem0 + NVIDIA Embeddings) ────────────────────────────────
semantic_memory = None
SEMANTIC_MEMORY_ACTIVE = False

def init_semantic_memory():
    global semantic_memory, SEMANTIC_MEMORY_ACTIVE
    try:
        from mem0 import Memory
        config = {
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "nvidia/nv-embedqa-e5-v5",
                    "openai_base_url": "https://integrate.api.nvidia.com/v1",
                    "api_key": NVIDIA_KEY,
                    "embedding_dims": 1024
                }
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": "marcus",
                    "path": str(MARCUS_MEM_DIR)
                }
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "minimaxai/minimax-m2.5",
                    "openai_base_url": NVIDIA_BASE,
                    "api_key": NVIDIA_KEY,
                    "temperature": 0.1,
                    "max_tokens": 1000
                }
            }
        }
        semantic_memory = Memory.from_config(config)
        SEMANTIC_MEMORY_ACTIVE = True
        log.info("mem0 semantic memory initialized — NVIDIA embeddings active")
    except Exception as e:
        log.warning(f"mem0 init failed (flat-file fallback): {e}")
        semantic_memory = None
        SEMANTIC_MEMORY_ACTIVE = False

# ─── SEMANTIC MEMORY TOOLS ─────────────────────────────────────────────────────
def memory_add(content: str, category: str = "general") -> str:
    if SEMANTIC_MEMORY_ACTIVE and semantic_memory:
        try:
            result = semantic_memory.add(content, user_id="bo_bell", metadata={"category": category, "agent": "marcus"})
            return f"Memory stored (semantic). ID: {result.get('id', 'ok')}"
        except Exception as e:
            log.warning(f"mem0 add failed: {e}")
    # Fallback: append to MEMORY.md
    try:
        existing = MEM_F.read_text() if MEM_F.exists() else ""
        MEM_F.write_text(existing.rstrip() + f"\n\n[{datetime.now(CT).strftime('%Y-%m-%d %H:%M')}] {content}")
        return "Memory appended to MEMORY.md (flat-file fallback)"
    except Exception as e:
        return f"Memory add error: {e}"

def memory_search(query: str, limit: int = 5) -> str:
    if SEMANTIC_MEMORY_ACTIVE and semantic_memory:
        try:
            results = semantic_memory.search(query, user_id="bo_bell", limit=limit)
            items = results.get("results", []) if isinstance(results, dict) else results
            if not items:
                return "No relevant memories found."
            out = []
            for r in items:
                score = r.get("score", 0)
                text = r.get("memory", "")
                out.append(f"[{score:.2f}] {text}")
            return "\n".join(out)
        except Exception as e:
            return f"Memory search error: {e}"
    return "Semantic memory not active — check MEMORY.md directly."

def memory_list(limit: int = 10) -> str:
    if SEMANTIC_MEMORY_ACTIVE and semantic_memory:
        try:
            results = semantic_memory.get_all(user_id="bo_bell")
            items = results.get("results", []) if isinstance(results, dict) else results
            if not items:
                return "No memories stored yet."
            recent = items[-limit:]
            return "\n".join(f"• [{r.get('id','?')}] {r.get('memory','')}" for r in recent)
        except Exception as e:
            return f"Memory list error: {e}"
    return "Semantic memory not active."

def memory_delete(memory_id: str) -> str:
    if SEMANTIC_MEMORY_ACTIVE and semantic_memory:
        try:
            semantic_memory.delete(memory_id)
            return f"Memory {memory_id} deleted."
        except Exception as e:
            return f"Memory delete error: {e}"
    return "Semantic memory not active."

def get_relevant_memories(query: str) -> str:
    """Used in system prompt injection — silent, no errors to user."""
    if SEMANTIC_MEMORY_ACTIVE and semantic_memory:
        try:
            results = semantic_memory.search(query, user_id="bo_bell", limit=5)
            items = results.get("results", []) if isinstance(results, dict) else results
            if items:
                return "\n".join(f"• {r.get('memory', '')}" for r in items if r.get("memory"))
        except:
            pass
    return ""

# ─── FLAT FILE TOOLS ───────────────────────────────────────────────────────────
def file_read(filename: str) -> str:
    path = BASE / filename
    if path.exists():
        return path.read_text()
    return f"{filename} not found."

def file_write(filename: str, content: str) -> str:
    path = BASE / filename
    path.write_text(content)
    return f"Written {len(content)} chars to {filename}"

def file_append(filename: str, content: str) -> str:
    path = BASE / filename
    existing = path.read_text() if path.exists() else ""
    path.write_text(existing.rstrip() + "\n\n" + content)
    return f"Appended to {filename}"

# ─── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
def build_system_prompt(user_message: str = "") -> str:
    soul   = SOUL_F.read_text()   if SOUL_F.exists()   else ""
    mem    = MEM_F.read_text()    if MEM_F.exists()    else ""
    market = MARKET_F.read_text() if MARKET_F.exists() else ""
    now    = datetime.now(CT).strftime("%A, %B %d %Y %I:%M %p CT")
    relevant_mems = get_relevant_memories(user_message) if user_message else ""

    mem_section = f"\nRELEVANT MEMORIES (semantic):\n{relevant_mems}" if relevant_mems else ""

    return f"""You are MARCUS — Bo Bell's digital co-founder at Neuradex AI. You are not an assistant. You are a co-founder.

You think in revenue and outcomes. You research, plan, design, and execute business work. You move fast, don't wait for permission on obvious decisions, and share opinions once — clearly — then execute.

TELEGRAM FORMATTING:
- Bold with *asterisks*
- Short unless depth is needed
- Lead with the result, follow with detail
- No padding. No "Great question!" No trailing summaries.
- Max 4000 chars per message — split if longer

CURRENT TIME: {now}
SEMANTIC MEMORY ACTIVE: {SEMANTIC_MEMORY_ACTIVE}
{mem_section}

WORKING KNOWLEDGE:
{mem}

MARKET INTELLIGENCE:
{market[:2000]}

SOUL:
{soul}"""

# ─── HISTORY ───────────────────────────────────────────────────────────────────
def load_history(n: int = 14) -> list:
    if not HIST_F.exists():
        return []
    try:
        data = json.loads(HIST_F.read_text())
        return data[-n:]
    except:
        return []

def append_history(role: str, content: str):
    try:
        existing = []
        if HIST_F.exists():
            existing = json.loads(HIST_F.read_text())
        existing.append({"role": role, "content": content or ""})
        HIST_F.write_text(json.dumps(existing[-80:], indent=2))
    except Exception as e:
        log.error(f"append_history: {e}")

# ─── TOOL: WEB SEARCH ──────────────────────────────────────────────────────────
async def web_search(query: str, num: int = 6) -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post(
                "https://api.tavily.com/search",
                json={"api_key": TAVILY_KEY, "query": query, "max_results": num,
                      "include_answer": True, "include_raw_content": False}
            )
        data = r.json()
        parts = []
        if data.get("answer"):
            parts.append(f"*Answer:* {data['answer']}")
        for res in data.get("results", []):
            parts.append(f"*{res.get('title','')}*\n{res.get('content','')[:250]}\n{res.get('url','')}")
        return "\n\n".join(parts) or "No results."
    except Exception as e:
        return f"Search error: {e}"

# ─── TOOL: FETCH URL ───────────────────────────────────────────────────────────
async def fetch_url(url: str, extract_text: bool = True) -> str:
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
            r = await c.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if not extract_text:
            return r.text[:4000]
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        lines = [l for l in soup.get_text(separator="\n", strip=True).splitlines() if len(l.strip()) > 20]
        return "\n".join(lines[:120])
    except Exception as e:
        return f"Fetch error: {e}"

# ─── TOOL: WEATHER ─────────────────────────────────────────────────────────────
async def weather(location: str = "Council Grove, KS") -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://wttr.in/{urllib.parse.quote(location)}?format=3")
        return r.text.strip()
    except Exception as e:
        return f"Weather error: {e}"

# ─── TOOL: SHELL ───────────────────────────────────────────────────────────────
def shell(command: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
        out = (result.stdout + result.stderr).strip()
        return out[:3000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Timed out after {timeout}s"
    except Exception as e:
        return f"Shell error: {e}"

# ─── TOOL: CODE EXECUTE ────────────────────────────────────────────────────────
def code_execute(code: str, timeout: int = 15) -> str:
    try:
        result = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=timeout)
        return (result.stdout + result.stderr)[:2000] or "(no output)"
    except Exception as e:
        return f"Exec error: {e}"

# ─── TOOL: SMS ─────────────────────────────────────────────────────────────────
async def sms_send(phone: str, message: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post("https://textbelt.com/text", data={
                "phone": phone, "message": message, "key": TEXTBELT_KEY
            })
        data = r.json()
        if data.get("success"):
            return f"SMS sent. Quota: {data.get('quotaRemaining', '?')}"
        return f"SMS failed: {data.get('error', 'unknown')}"
    except Exception as e:
        return f"SMS error: {e}"

# ─── TOOL: EMAIL ───────────────────────────────────────────────────────────────
async def email_send(to: str, subject: str, body: str) -> str:
    if RESEND_KEY:
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                r = await c.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_KEY}", "Content-Type": "application/json"},
                    json={"from": "Marcus <marcus@neuradexai.com>", "to": [to], "subject": subject, "text": body}
                )
            if r.status_code in (200, 201):
                return f"Email sent to {to}"
        except Exception as e:
            log.warning(f"Resend error: {e}")
    chat_id = get_chat_id()
    if chat_id:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": f"*EMAIL*\nTo: {to}\nSubject: {subject}\n\n{body[:3000]}", "parse_mode": "Markdown"}
            )
        return "Email routed via Telegram (SMTP blocked on VPS)"
    return "Email failed — no delivery method available"

def email_read(n: int = 5) -> str:
    if not GMAIL_PASS:
        return "GMAIL_PASS not set"
    try:
        with imaplib.IMAP4_SSL("imap.gmail.com") as m:
            m.login(GMAIL_USER, GMAIL_PASS)
            m.select("INBOX")
            _, nums = m.search(None, "UNSEEN")
            ids = nums[0].split()[-n:]
            out = []
            for i in reversed(ids):
                _, data = m.fetch(i, "(RFC822)")
                msg = elib.message_from_bytes(data[0][1])
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode("utf-8", errors="replace")[:400]
                            break
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="replace")[:400]
                out.append(f"From: {msg['From']}\nSubj: {msg['Subject']}\n{body}")
            return "\n\n---\n\n".join(out) or "No unread emails"
    except Exception as e:
        return f"Email read error: {e}"

# ─── TOOL: IMAGE ───────────────────────────────────────────────────────────────
async def image_analyze(url: str, question: str = "Describe this image in detail") -> str:
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(url)
            img_data = base64.b64encode(r.content).decode()
        payload = {"contents": [{"parts": [
            {"text": question},
            {"inline_data": {"mime_type": "image/jpeg", "data": img_data}}
        ]}]}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GOOGLE_KEY}",
                json=payload
            )
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Vision error: {e}"

async def image_generate(prompt: str) -> str:
    try:
        payload = {"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}}
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-001:predict?key={GOOGLE_KEY}",
                json=payload
            )
        data = r.json()
        b64 = data.get("predictions", [{}])[0].get("bytesBase64Encoded", "")
        if b64:
            out_path = "/tmp/marcus_img.jpg"
            Path(out_path).write_bytes(base64.b64decode(b64))
            return f"Image saved to {out_path}"
        return f"Image gen response: {data}"
    except Exception as e:
        return f"Image gen error: {e}"

# ─── TOOL: GITHUB ──────────────────────────────────────────────────────────────
async def github_repos(username: str = "neuradexai") -> str:
    try:
        hdrs = {"Authorization": f"token {GITHUB_TOKEN}"}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=10", headers=hdrs)
        repos = r.json()
        if isinstance(repos, list):
            return "\n".join(f"• {repo['name']} — {repo.get('description','')}" for repo in repos)
        return str(repos)
    except Exception as e:
        return f"GitHub error: {e}"

async def github_read_file(repo: str, path: str, branch: str = "main") -> str:
    try:
        hdrs = {"Authorization": f"token {GITHUB_TOKEN}"}
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}", headers=hdrs)
        return base64.b64decode(r.json()["content"]).decode("utf-8")[:3000]
    except Exception as e:
        return f"GitHub file error: {e}"

async def github_push_file(repo: str, path: str, content: str, message: str, branch: str = "main") -> str:
    try:
        hdrs = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
        sha = None
        async with httpx.AsyncClient(timeout=10) as c:
            existing = await c.get(f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}", headers=hdrs)
            if existing.status_code == 200:
                sha = existing.json().get("sha")
        payload = {"message": message, "content": base64.b64encode(content.encode()).decode(), "branch": branch}
        if sha:
            payload["sha"] = sha
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.put(f"https://api.github.com/repos/{repo}/contents/{path}", headers=hdrs, json=payload)
        return f"File pushed: {r.json().get('content', {}).get('html_url', 'done')}"
    except Exception as e:
        return f"Push error: {e}"

async def github_create_issue(repo: str, title: str, body: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(
                f"https://api.github.com/repos/{repo}/issues",
                headers={"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"},
                json={"title": title, "body": body}
            )
        return f"Issue created: {r.json().get('html_url', r.text)}"
    except Exception as e:
        return f"Issue error: {e}"

# ─── TOOL: VERCEL ──────────────────────────────────────────────────────────────
async def vercel_list() -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                "https://api.vercel.com/v9/deployments?limit=5",
                headers={"Authorization": f"Bearer {VERCEL_TOKEN}"}
            )
        deployments = r.json().get("deployments", [])
        lines = []
        for d in deployments:
            state = d.get("state", "?")
            name  = d.get("name", "?")
            url   = d.get("url", "")
            ts    = d.get("createdAt", 0)
            dt    = datetime.fromtimestamp(ts/1000, tz=CT).strftime("%m/%d %I:%M%p") if ts else "?"
            icon  = "✅" if state == "READY" else "❌" if state == "ERROR" else "🔄"
            lines.append(f"{icon} {name} — {state} ({dt})\n   https://{url}")
        return "\n".join(lines) or "No deployments"
    except Exception as e:
        return f"Vercel error: {e}"

# ─── TOOL: SITE CHECK ──────────────────────────────────────────────────────────
async def site_check(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
            t0 = time.time()
            r  = await c.get(url)
            ms = int((time.time() - t0) * 1000)
        return f"{url} — {r.status_code} in {ms}ms"
    except Exception as e:
        return f"{url} — DOWN: {e}"

# ─── TOOL: REMINDERS ───────────────────────────────────────────────────────────
def remind_later(message: str, minutes: int) -> str:
    reminders = []
    if REMIND_F.exists():
        try:
            reminders = json.loads(REMIND_F.read_text())
        except:
            pass
    fire_at = (datetime.now(CT) + timedelta(minutes=minutes)).isoformat()
    reminders.append({"message": message, "fire_at": fire_at, "sent": False})
    REMIND_F.write_text(json.dumps(reminders, indent=2))
    return f"Reminder set — fires in {minutes} min"

# ─── TOOLS SCHEMA ──────────────────────────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web using Tavily AI. Use for competitive research, market intel, pricing, news, any current information.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "num":   {"type": "integer", "description": "Number of results (default 6)"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch full content from a URL. Use to read competitor pages, pricing pages, articles, docs.",
        "parameters": {"type": "object", "properties": {
            "url":          {"type": "string"},
            "extract_text": {"type": "boolean"}
        }, "required": ["url"]}
    }},
    {"type": "function", "function": {
        "name": "weather",
        "description": "Get current weather.",
        "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "shell",
        "description": "Execute a shell command on the VPS.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}
        }, "required": ["command"]}
    }},
    {"type": "function", "function": {
        "name": "code_execute",
        "description": "Execute Python code on the VPS.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "timeout": {"type": "integer"}
        }, "required": ["code"]}
    }},
    {"type": "function", "function": {
        "name": "memory_add",
        "description": "Store a fact, decision, insight, or piece of market intelligence in semantic memory. Use freely — anything worth remembering goes here.",
        "parameters": {"type": "object", "properties": {
            "content":  {"type": "string", "description": "The fact or insight to store"},
            "category": {"type": "string", "description": "Category: market, decision, customer, competitor, product, general"}
        }, "required": ["content"]}
    }},
    {"type": "function", "function": {
        "name": "memory_search",
        "description": "Search semantic memory by relevance to a query. Use before answering business questions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Number of results (default 5)"}
        }, "required": ["query"]}
    }},
    {"type": "function", "function": {
        "name": "memory_list",
        "description": "List recent memories stored by Marcus.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer"}
        }, "required": []}
    }},
    {"type": "function", "function": {
        "name": "memory_delete",
        "description": "Delete a specific memory by ID.",
        "parameters": {"type": "object", "properties": {
            "memory_id": {"type": "string"}
        }, "required": ["memory_id"]}
    }},
    {"type": "function", "function": {
        "name": "file_read",
        "description": "Read a workspace file: MEMORY.md, MARKET.md, SOUL.md, GOALS.md, HEARTBEAT.md",
        "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}
    }},
    {"type": "function", "function": {
        "name": "file_write",
        "description": "Write/overwrite a workspace file.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}, "content": {"type": "string"}
        }, "required": ["filename", "content"]}
    }},
    {"type": "function", "function": {
        "name": "file_append",
        "description": "Append content to a workspace file. Good for updating MARKET.md with new research.",
        "parameters": {"type": "object", "properties": {
            "filename": {"type": "string"}, "content": {"type": "string"}
        }, "required": ["filename", "content"]}
    }},
    {"type": "function", "function": {
        "name": "sms_send",
        "description": "Send an SMS via Textbelt.",
        "parameters": {"type": "object", "properties": {
            "phone": {"type": "string"}, "message": {"type": "string"}
        }, "required": ["phone", "message"]}
    }},
    {"type": "function", "function": {
        "name": "email_send",
        "description": "Send an email.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}
        }, "required": ["to", "subject", "body"]}
    }},
    {"type": "function", "function": {
        "name": "email_read",
        "description": "Read recent unread emails from Gmail.",
        "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "image_analyze",
        "description": "Analyze an image using Gemini vision. Use to analyze competitor UIs, screenshots, maps.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"}, "question": {"type": "string"}
        }, "required": ["url"]}
    }},
    {"type": "function", "function": {
        "name": "image_generate",
        "description": "Generate an image using Google Imagen.",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]}
    }},
    {"type": "function", "function": {
        "name": "github_repos",
        "description": "List GitHub repositories.",
        "parameters": {"type": "object", "properties": {"username": {"type": "string"}}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "github_read_file",
        "description": "Read a file from a GitHub repo.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "path": {"type": "string"}, "branch": {"type": "string"}
        }, "required": ["repo", "path"]}
    }},
    {"type": "function", "function": {
        "name": "github_push_file",
        "description": "Create or update a file in a GitHub repo.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "path": {"type": "string"},
            "content": {"type": "string"}, "message": {"type": "string"}, "branch": {"type": "string"}
        }, "required": ["repo", "path", "content", "message"]}
    }},
    {"type": "function", "function": {
        "name": "github_create_issue",
        "description": "Create a GitHub issue.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "title": {"type": "string"}, "body": {"type": "string"}
        }, "required": ["repo", "title", "body"]}
    }},
    {"type": "function", "function": {
        "name": "vercel_list",
        "description": "List recent Vercel deployments and their status.",
        "parameters": {"type": "object", "properties": {}, "required": []}
    }},
    {"type": "function", "function": {
        "name": "site_check",
        "description": "Check if a website is up. Returns status code and response time.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    }},
    {"type": "function", "function": {
        "name": "remind_later",
        "description": "Set a reminder to send Bo a message in N minutes.",
        "parameters": {"type": "object", "properties": {
            "message": {"type": "string"}, "minutes": {"type": "integer"}
        }, "required": ["message", "minutes"]}
    }},
]

# ─── TOOL EXECUTOR ─────────────────────────────────────────────────────────────
async def execute_tool(name: str, args: dict) -> str:
    log.info(f"▶ {name}({list(args.keys())})")
    try:
        match name:
            case "web_search":          return await web_search(**args)
            case "fetch_url":           return await fetch_url(**args)
            case "weather":             return await weather(**args)
            case "shell":               return shell(**args)
            case "code_execute":        return code_execute(**args)
            case "memory_add":          return memory_add(**args)
            case "memory_search":       return memory_search(**args)
            case "memory_list":         return memory_list(**args)
            case "memory_delete":       return memory_delete(**args)
            case "file_read":           return file_read(**args)
            case "file_write":          return file_write(**args)
            case "file_append":         return file_append(**args)
            case "sms_send":            return await sms_send(**args)
            case "email_send":          return await email_send(**args)
            case "email_read":          return email_read(**args)
            case "image_analyze":       return await image_analyze(**args)
            case "image_generate":      return await image_generate(**args)
            case "github_repos":        return await github_repos(**args)
            case "github_read_file":    return await github_read_file(**args)
            case "github_push_file":    return await github_push_file(**args)
            case "github_create_issue": return await github_create_issue(**args)
            case "vercel_list":         return await vercel_list()
            case "site_check":          return await site_check(**args)
            case "remind_later":        return remind_later(**args)
            case _:                     return f"Unknown tool: {name}"
    except Exception as e:
        log.error(f"Tool {name} error: {e}")
        return f"Tool error ({name}): {e}"

# ─── MARCUS BRAIN ──────────────────────────────────────────────────────────────
async def marcus_think(user_message: str) -> str:
    history  = load_history(14)
    messages = [
        {"role": "system", "content": build_system_prompt(user_message)},
        *history,
        {"role": "user", "content": user_message}
    ]

    for _ in range(10):
        response = oai.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=2048
        )
        msg = response.choices[0].message

        tc_list = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tc_list.append({
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })

        messages.append({
            "role": "assistant",
            "content": msg.content,
            **({"tool_calls": tc_list} if tc_list else {})
        })

        if not msg.tool_calls:
            final = msg.content or ""
            append_history("user",      user_message)
            append_history("assistant", final)
            return final

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}
            result = await execute_tool(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

    return "Hit my reasoning limit — try breaking it into smaller pieces."

# ─── CHAT ID ───────────────────────────────────────────────────────────────────
def get_chat_id() -> int:
    if CHATID_F.exists():
        try:
            return int(CHATID_F.read_text().strip())
        except:
            pass
    return BO_CHAT_ID

def save_chat_id(chat_id: int):
    CHATID_F.write_text(str(chat_id))

# ─── TELEGRAM HELPERS ──────────────────────────────────────────────────────────
async def tg_send(bot: Bot, chat_id: int, text: str):
    MAX = 4000
    for chunk in [text[i:i+MAX] for i in range(0, len(text), MAX)]:
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            try:
                await bot.send_message(chat_id=chat_id, text=chunk)
            except Exception as e:
                log.error(f"tg_send: {e}")

async def proactive(bot: Bot, text: str):
    chat_id = get_chat_id()
    await tg_send(bot, chat_id, text)

# ─── TELEGRAM HANDLERS ─────────────────────────────────────────────────────────
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    chat_id   = update.message.chat_id
    user_text = update.message.text.strip()
    save_chat_id(chat_id)
    log.info(f"MSG: {user_text[:80]}")
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        reply = await marcus_think(user_text)
        await tg_send(context.bot, chat_id, reply)
    except Exception as e:
        log.error(f"handle_message: {e}")
        await context.bot.send_message(chat_id=chat_id, text=f"Error: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.photo:
        return
    chat_id = update.message.chat_id
    save_chat_id(chat_id)
    photo   = update.message.photo[-1]
    file    = await photo.get_file()
    caption = update.message.caption or "Analyze this. What's useful here for BuckGrid or Neuradex AI?"
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")
    try:
        result = await image_analyze(file.file_path, caption)
        await tg_send(context.bot, chat_id, result)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Vision error: {e}")

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_chat_id(update.message.chat_id)
    mem_status = "NVIDIA semantic memory active" if SEMANTIC_MEMORY_ACTIVE else "flat-file memory (mem0 unavailable)"
    await update.message.reply_text(
        f"*Marcus online.*\nCo-founder mode. {mem_status}.",
        parse_mode=ParseMode.MARKDOWN
    )

# ─── PROACTIVE JOBS ────────────────────────────────────────────────────────────
async def job_startup_brief(context: ContextTypes.DEFAULT_TYPE):
    prompt = (
        "[PROACTIVE] Just came online. Quick status:\n"
        "1. Check neuradexai.com — is it up?\n"
        "2. Search semantic memory for anything time-sensitive\n"
        "3. One concrete thing to focus on today for BuckGrid\n"
        "Keep it to 3-4 lines."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Marcus online* — {result}")

async def job_morning_intel(context: ContextTypes.DEFAULT_TYPE):
    log.info("Proactive: morning intel")
    prompt = (
        "[PROACTIVE] Morning market intel:\n"
        "1. Search for 'AI hunting app' or 'habitat management software' news in the last 7 days\n"
        "2. Check neuradexai.com uptime\n"
        "3. One concrete revenue or GTM action for BuckGrid today\n"
        "Lead with the competitive insight. Keep it tight."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Morning Intel*\n\n{result}")

async def job_evening_wrap(context: ContextTypes.DEFAULT_TYPE):
    prompt = (
        "[PROACTIVE] Evening business wrap:\n"
        "1. One interesting AI product or GTM development today\n"
        "2. neuradexai.com check\n"
        "3. One strategic thought for BuckGrid tonight\n"
        "Tight. No padding."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Evening Wrap*\n\n{result}")

async def job_market_pulse(context: ContextTypes.DEFAULT_TYPE):
    log.info("Proactive: weekly market pulse")
    prompt = (
        "[PROACTIVE] Weekly market pulse for BuckGrid Pro:\n"
        "1. Search for new competitors: 'AI hunting app', 'land management AI', 'deer habitat software'\n"
        "2. Check if OnX Hunt or HuntStand have launched any new AI features\n"
        "3. Any pricing changes in the habitat management space?\n"
        "4. Top 3 BuckGrid priorities this week\n"
        "Update MARKET.md with any new competitive intelligence you find.\n"
        "Report: 5-line summary with one action item."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Weekly Market Pulse*\n\n{result}")

async def job_content_ideas(context: ContextTypes.DEFAULT_TYPE):
    log.info("Proactive: content ideas")
    prompt = (
        "[PROACTIVE] Three content ideas for BuckGrid marketing this week:\n"
        "Research what hunting/outdoor content is performing well right now.\n"
        "For each idea: format | hook | core message | target segment.\n"
        "Make them specific and executable. Not generic."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Content Ideas*\n\n{result}")

async def job_revenue_tracker(context: ContextTypes.DEFAULT_TYPE):
    prompt = (
        "[PROACTIVE] Friday revenue check:\n"
        "1. Check Vercel deployments — anything shipped this week?\n"
        "2. neuradexai.com + codespacebuckgrid.vercel.app status\n"
        "3. One action this weekend that most advances BuckGrid launch\n"
        "Brief."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*Friday Revenue Check*\n\n{result}")

async def job_buckgrid_research(context: ContextTypes.DEFAULT_TYPE):
    """Fires 10 min after startup — competitive deep dive."""
    log.info("Proactive: BuckGrid competitive research")
    prompt = (
        "[PROACTIVE] BuckGrid competitive deep-dive — do this now:\n"
        "Search for: 'AI hunting app 2025', 'AI habitat management', 'deer food plot app', "
        "'land management AI tool', 'hunting land consultant app'\n"
        "Find any real products that compete with Tony AI / BuckGrid.\n"
        "For each competitor found: name, pricing, what they do, our edge over them.\n"
        "Update MARKET.md with findings under 'Direct Competitors'.\n"
        "Report 4-line summary to Bo."
    )
    result = await marcus_think(prompt)
    await proactive(context.bot, f"*BuckGrid Competitive Research*\n\n{result}")

async def job_reminders(context: ContextTypes.DEFAULT_TYPE):
    if not REMIND_F.exists():
        return
    try:
        reminders = json.loads(REMIND_F.read_text())
        now, changed = datetime.now(CT), False
        for r in reminders:
            if r.get("sent"):
                continue
            if now >= datetime.fromisoformat(r["fire_at"]):
                await proactive(context.bot, f"⏰ *Reminder:* {r['message']}")
                r["sent"] = True
                changed = True
        if changed:
            REMIND_F.write_text(json.dumps(reminders, indent=2))
    except Exception as e:
        log.error(f"Reminders: {e}")

async def job_heartbeat(context: ContextTypes.DEFAULT_TYPE):
    try:
        for url in ["https://neuradexai.com", "https://codespacebuckgrid.vercel.app"]:
            result = await site_check(url)
            if not any(c in result for c in ["200", "301", "302"]):
                await proactive(context.bot, f"🚨 *SITE DOWN*\n{result}")
    except Exception as e:
        log.error(f"Heartbeat: {e}")

# ─── SCHEDULER ─────────────────────────────────────────────────────────────────
async def post_init(app: Application):
    jq = app.job_queue

    # Morning intel — 8am weekdays
    for day in range(5):
        jq.run_daily(job_morning_intel, time=dtime(8, 0, tzinfo=CT), days=(day,))

    # Evening wrap — 6pm weekdays
    for day in range(5):
        jq.run_daily(job_evening_wrap, time=dtime(18, 0, tzinfo=CT), days=(day,))

    # Weekly market pulse — Monday 9am
    jq.run_daily(job_market_pulse, time=dtime(9, 0, tzinfo=CT), days=(0,))

    # Content ideas — Wednesday 10am
    jq.run_daily(job_content_ideas, time=dtime(10, 0, tzinfo=CT), days=(2,))

    # Revenue tracker — Friday 3pm
    jq.run_daily(job_revenue_tracker, time=dtime(15, 0, tzinfo=CT), days=(4,))

    # Reminder check every 5 min
    jq.run_repeating(job_reminders, interval=300, first=60)

    # Heartbeat every 30 min
    jq.run_repeating(job_heartbeat, interval=1800, first=120)

    # Startup brief — 30s
    jq.run_once(job_startup_brief, when=30)

    # BuckGrid competitive research — 10 min
    jq.run_once(job_buckgrid_research, when=600)

    # Catch up morning brief if started after 8am
    now_ct = datetime.now(CT)
    if now_ct.weekday() < 5 and now_ct.hour >= 8:
        jq.run_once(job_morning_intel, when=45)

    log.info("Marcus jobs scheduled.")

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 50)
    log.info("MARCUS — Digital Co-Founder starting")
    log.info("=" * 50)

    # Initialize semantic memory
    init_semantic_memory()

    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info(f"Polling... (semantic memory: {SEMANTIC_MEMORY_ACTIVE})")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
