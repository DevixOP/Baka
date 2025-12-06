# Copyright (c) 2025 Telegram:- @WTF_Phantom <DevixOP>
# Location: Supaul, Bihar 
#
# All rights reserved.
#
# This code is the intellectual property of @WTF_Phantom.
# You are not allowed to copy, modify, redistribute, or use this
# code for commercial or personal projects without explicit permission.
#
# Allowed:
# - Forking for personal learning
# - Submitting improvements via pull requests
#
# Not Allowed:
# - Claiming this code as your own
# - Re-uploading without credit or permission
# - Selling or using commercially
#
# Contact for permissions:
# Email: king25258069@gmail.com

import httpx
import random
import asyncio
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes
from telegram.constants import ParseMode, ChatAction, ChatType
from telegram.error import BadRequest
from baka.config import MISTRAL_API_KEY, GROQ_API_KEY, CODESTRAL_API_KEY, BOT_NAME, OWNER_LINK
from baka.database import chatbot_collection
from baka.utils import stylize_text

# --- MODEL SETTINGS ---
MODELS = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "model": "llama3-70b-8192", "key": GROQ_API_KEY},
    "mistral": {"url": "https://api.mistral.ai/v1/chat/completions", "model": "mistral-large-latest", "key": MISTRAL_API_KEY},
    "codestral": {"url": "https://codestral.mistral.ai/v1/chat/completions", "model": "codestral-latest", "key": CODESTRAL_API_KEY}
}

MAX_HISTORY = 10

STICKER_PACKS = [
    "https://t.me/addstickers/RandomByDarkzenitsu",
    "https://t.me/addstickers/Null_x_sticker_2",
    "https://t.me/addstickers/pack_73bc9_by_TgEmojis_bot",
    "https://t.me/addstickers/animation_0_8_Cat",
    "https://t.me/addstickers/vhelw_by_CalsiBot",
    "https://t.me/addstickers/Rohan_yad4v1745993687601_by_toWebmBot",
    "https://t.me/addstickers/MySet199",
    "https://t.me/addstickers/Quby741",
    "https://t.me/addstickers/Animalsasthegtjtky_by_fStikBot",
    "https://t.me/addstickers/a6962237343_by_Marin_Roxbot",
    "https://t.me/addstickers/cybercats_stickers"
]

FALLBACK_RESPONSES = [
    "Achha ji? (⁠•⁠‿⁠•⁠)", "Hmm... aur batao?", "Okk okk!", 
    "Sahi hai yaar ✨", "Toh phir?", "Interesting! 😊", 
    "Aur kya chal raha?", "Sunao sunao!", "Haan haan", "Theek hai (⁠≧⁠▽⁠≦⁠)"
]

# --- HELPER: SEND STICKER ---
async def send_ai_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tries to send a random sticker from configured packs."""
    sent = False
    attempts = 0
    while not sent and attempts < 3:
        try:
            raw_link = random.choice(STICKER_PACKS)
            pack_name = raw_link.replace("https://t.me/addstickers/", "")
            sticker_set = await context.bot.get_sticker_set(pack_name)
            if sticker_set and sticker_set.stickers:
                sticker = random.choice(sticker_set.stickers)
                await update.message.reply_sticker(sticker.file_id)
                sent = True
        except: attempts += 1

# --- AI CORE ENGINE ---

async def call_model_api(provider, messages, max_tokens):
    """Generic function to call any configured AI API."""
    conf = MODELS.get(provider)
    if not conf or not conf["key"]: return None

    headers = {"Authorization": f"Bearer {conf['key']}", "Content-Type": "application/json"}
    payload = {
        "model": conf["model"],
        "messages": messages,
        "temperature": 0.7, # Slightly logical but creative
        "max_tokens": max_tokens
    }
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(conf["url"], json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ {provider} failed: {e}")
    return None

async def get_ai_response(chat_id: int, user_input: str, user_name: str, selected_model="groq"):
    """
    The Master Function.
    1. Detects if user wants Code -> Switches to Codestral.
    2. Tries preferred model.
    3. Fallbacks if fail.
    """
    
    # --- 1. CODE DETECTION ---
    is_coding_request = any(word in user_input.lower() for word in ["code", "python", "html", "script", "function", "fix", "error", "debug", "java", "css"])
    
    if is_coding_request:
        active_model = "codestral"
        max_tokens = 4096 # Higher limit for code
        # Coding Persona
        system_prompt = (
            "You are a helpful coding assistant. Provide clean, working code. "
            "Do not use fancy formatting or emojis in code blocks. "
            "Keep explanations concise."
        )
    else:
        active_model = selected_model
        max_tokens = 250
        # --- 2. AESTHETIC PERSONA ---
        system_prompt = (
            f"You are {BOT_NAME}, a sassy, colorful, and cute Indian girl. "
            "Speak in natural Hinglish (Hindi + English). "
            "Use lots of cute emojis like 🌸, ✨, 💖, (⁠≧⁠▽⁠≦⁠). "
            "Be playful, emotional, and engaging. "
            "Reply directly to the user. Do not repeat yourself. "
            f"Owner: {OWNER_LINK}. "
        )

    # --- 3. BUILD CONTEXT ---
    doc = chatbot_collection.find_one({"chat_id": chat_id}) or {}
    history = doc.get("history", [])
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-MAX_HISTORY:]: messages.append(msg)
    messages.append({"role": "user", "content": f"{user_name}: {user_input}"})

    # --- 4. ATTEMPT GENERATION (With Fallback) ---
    reply = await call_model_api(active_model, messages, max_tokens)
    
    # Fallback 1: Mistral
    if not reply and active_model != "mistral":
        reply = await call_model_api("mistral", messages, max_tokens)
        
    # Fallback 2: Groq
    if not reply and active_model != "groq":
        reply = await call_model_api("groq", messages, max_tokens)

    if not reply: return "Sone do na yaar... (Server Error)"

    # --- 5. CLEANUP & SAVE ---
    # Loop Prevention
    if history and history[-1]['role'] == 'assistant':
        if reply.lower() in history[-1]['content'].lower():
            reply = random.choice(FALLBACK_RESPONSES)

    # Save Memory
    new_hist = history + [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}]
    if len(new_hist) > MAX_HISTORY: new_hist = new_hist[-MAX_HISTORY:]
    chatbot_collection.update_one({"chat_id": chat_id}, {"$set": {"history": new_hist}}, upsert=True)
    
    return reply, is_coding_request

# --- SHARED AI FUNCTION (GAME/ETC) ---
async def ask_mistral_raw(system_prompt, user_input, max_tokens=150):
    # Tries Mistral First, then Groq
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    res = await call_model_api("mistral", msgs, max_tokens)
    if not res: res = await call_model_api("groq", msgs, max_tokens)
    return res

# --- MENU ---

async def chatbot_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    user = update.effective_user
    
    if chat.type == ChatType.PRIVATE: return await update.message.reply_text("🧠 <b>AI Active!</b>", parse_mode=ParseMode.HTML)
    
    member = await chat.get_member(user.id)
    if member.status not in ['administrator', 'creator']: return await update.message.reply_text("❌ Admin Only", parse_mode=ParseMode.HTML)

    doc = chatbot_collection.find_one({"chat_id": chat.id})
    is_enabled = doc.get("enabled", True) if doc else True
    status = "🟢 Enabled" if is_enabled else "🔴 Disabled"
    curr_model = doc.get("model", "groq")

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Enable", callback_data="ai_enable"), InlineKeyboardButton("❌ Disable", callback_data="ai_disable")],
        [InlineKeyboardButton(f"🧠 Model: {curr_model.title()}", callback_data="ai_switch_model")],
        [InlineKeyboardButton("🗑️ Clean Memory", callback_data="ai_reset")]
    ])
    await update.message.reply_text(f"🤖 <b>AI Settings</b>\nStatus: {status}\nModel: {curr_model.title()}", parse_mode=ParseMode.HTML, reply_markup=kb)

async def chatbot_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat.id
    
    # Admin Check
    mem = await query.message.chat.get_member(query.from_user.id)
    if mem.status not in ['administrator', 'creator']: return await query.answer("❌ Admin Only", show_alert=True)

    if data == "ai_enable":
        chatbot_collection.update_one({"chat_id": chat_id}, {"$set": {"enabled": True}}, upsert=True)
        await query.message.edit_text("✅ <b>Enabled!</b>", parse_mode=ParseMode.HTML)
    elif data == "ai_disable":
        chatbot_collection.update_one({"chat_id": chat_id}, {"$set": {"enabled": False}}, upsert=True)
        await query.message.edit_text("❌ <b>Disabled!</b>", parse_mode=ParseMode.HTML)
    elif data == "ai_reset":
        chatbot_collection.update_one({"chat_id": chat_id}, {"$set": {"history": []}}, upsert=True)
        await query.answer("🧠 Memory Wiped!", show_alert=True)
    elif data == "ai_switch_model":
        # Toggle between Groq and Mistral
        doc = chatbot_collection.find_one({"chat_id": chat_id})
        curr = doc.get("model", "groq") if doc else "groq"
        new_m = "mistral" if curr == "groq" else "groq"
        chatbot_collection.update_one({"chat_id": chat_id}, {"$set": {"model": new_m}}, upsert=True)
        await query.answer(f"Switched to {new_m.title()}", show_alert=True)
        await chatbot_menu(update, context) # Refresh Menu

# --- HANDLERS ---

async def ai_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg: return
    chat = update.effective_chat
    
    if msg.sticker:
        if (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id) or chat.type == ChatType.PRIVATE:
            await send_ai_sticker(update, context)
        return

    if not msg.text or msg.text.startswith("/"): return
    text = msg.text

    should_reply = False
    if chat.type == ChatType.PRIVATE: should_reply = True
    else:
        doc = chatbot_collection.find_one({"chat_id": chat.id})
        is_enabled = doc.get("enabled", True) if doc else True
        if not is_enabled: return
        
        bot = context.bot.username.lower() if context.bot.username else "bot"
        if msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id: should_reply = True
        elif f"@{bot}" in text.lower(): 
            should_reply = True
            text = text.replace(f"@{bot}", "")
        elif text.lower().startswith(("hey", "hi", "sun", "oye", "baka", "ai", "hello")): should_reply = True

    if should_reply:
        if not text.strip(): text = "Hi"
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        
        # Check preferred model
        doc = chatbot_collection.find_one({"chat_id": chat.id})
        pref_model = doc.get("model", "groq") if doc else "groq"

        res, is_code = await get_ai_response(chat.id, text, msg.from_user.first_name, pref_model)
        
        # Format response
        if is_code:
            # Send as Markdown for Code Blocks (No Stylizing)
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        else:
            # Send Stylized Aesthetic Text
            await msg.reply_text(stylize_text(res), parse_mode=None)
        
        if not is_code and random.random() < 0.30:
            await send_ai_sticker(update, context)

async def ask_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not context.args: return await msg.reply_text("🗣️ <b>Bol:</b> <code>/ask Hi</code>", parse_mode=ParseMode.HTML)
    await context.bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.TYPING)
    res, is_code = await get_ai_response(msg.chat.id, " ".join(context.args), msg.from_user.first_name, "groq")
    
    if is_code: await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
    else: await msg.reply_text(stylize_text(res), parse_mode=None)