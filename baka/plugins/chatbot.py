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

# --- 🎨 BAKA PERSONALITY CONFIG ---
BAKA_NAME = "Baka"
BAKA_TRAITS = [
    "playful Indian girlfriend",
    "sassy but sweet",
    "uses cute Hinglish",
    "expressive with emojis",
    "keeps replies short (1-2 lines) unless coding",
    "emotionally intelligent",
    "never repeats phrases back-to-back"
]

# Rotating emoji pools (fresh every response)
EMOJI_POOL = ["✨", "💖", "🌸", "😊", "🥰", "💕", "🎀", "🌺", "💫", "🦋", "🌼", "💗", "🎨", "🍓"]

# --- 🤖 MODEL SETTINGS ---
MODELS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama3-70b-8192",
        "key": GROQ_API_KEY
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "model": "mistral-large-latest",
        "key": MISTRAL_API_KEY
    },
    "codestral": {
        "url": "https://codestral.mistral.ai/v1/chat/completions",
        "model": "codestral-latest",
        "key": CODESTRAL_API_KEY
    }
}

MAX_HISTORY = 10
DEFAULT_MODEL = "mistral"  # Changed from groq

# --- 🎭 STICKER PACKS ---
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
    f"Achha ji? {random.choice(EMOJI_POOL)}",
    "Hmm... aur batao?",
    f"Okk okk! {random.choice(EMOJI_POOL)}",
    f"Sahi hai yaar {random.choice(EMOJI_POOL)}",
    "Toh phir?",
    f"Interesting! {random.choice(EMOJI_POOL)}",
    "Aur kya chal raha?",
    f"Sunao sunao! {random.choice(EMOJI_POOL)}",
    "Haan haan",
    f"Theek hai {random.choice(EMOJI_POOL)}"
]

# --- 📨 HELPER: SEND STICKER ---
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
        except:
            attempts += 1

# --- 🧠 AI CORE ENGINE ---

async def call_model_api(provider, messages, max_tokens):
    """Generic function to call any configured AI API."""
    conf = MODELS.get(provider)
    if not conf or not conf["key"]:
        return None

    headers = {
        "Authorization": f"Bearer {conf['key']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": conf["model"],
        "messages": messages,
        "temperature": 0.75,  # Balanced creativity
        "max_tokens": max_tokens
    }

    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(conf["url"], json=payload, headers=headers)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"⚠️ {provider} API failed: {e}")
    return None


async def get_ai_response(chat_id: int, user_input: str, user_name: str, selected_model=DEFAULT_MODEL):
    """
    🎯 The Master AI Function
    
    Flow:
    1. Detects if user wants code → Auto-switches to Codestral
    2. Uses user's preferred model (default: Mistral)
    3. Fallback chain: Selected → Mistral → Groq
    4. Anti-loop protection
    5. Context-aware responses
    """

    # --- 1️⃣ CODE DETECTION ---
    code_keywords = [
        "code", "python", "html", "css", "javascript", "script", 
        "function", "fix", "error", "debug", "java", "algorithm",
        "program", "syntax", "class", "import", "def ", "npm"
    ]
    is_coding_request = any(kw in user_input.lower() for kw in code_keywords)

    if is_coding_request:
        active_model = "codestral"
        max_tokens = 4096
        # 🖥️ Codestral Persona (Technical, Clean)
        system_prompt = (
            "You are a professional coding assistant. "
            "Provide clean, working, well-commented code. "
            "Explain briefly but precisely. No emojis in code blocks. "
            "Support Python, JavaScript, HTML, CSS, Java, C++."
        )
    else:
        active_model = selected_model
        max_tokens = 200  # Short replies for conversation
        # 💕 Baka Persona (Sassy Girlfriend)
        emoji_set = random.sample(EMOJI_POOL, 3)  # Pick 3 random emojis
        system_prompt = (
            f"You are {BAKA_NAME}, a playful Indian girlfriend who talks in natural Hinglish. "
            f"Personality: Sassy, sweet, emotionally expressive, uses emojis like {' '.join(emoji_set)}. "
            "Rules:\n"
            "- Reply in 1-2 sentences max (unless user asks for details)\n"
            "- Mix Hindi and English naturally (e.g., 'Kya hua baby?', 'Achha theek hai ✨')\n"
            "- Never repeat the same phrase twice in a row\n"
            "- Be direct and engaging, not robotic\n"
            "- Show emotions: excitement, teasing, care, playfulness\n"
            f"- Owner: {OWNER_LINK}\n"
            "- Don't mention you're an AI unless asked"
        )

    # --- 2️⃣ BUILD CONTEXT ---
    doc = chatbot_collection.find_one({"chat_id": chat_id}) or {}
    history = doc.get("history", [])

    messages = [{"role": "system", "content": system_prompt}]
    
    # Add recent context (last 10 exchanges)
    for msg in history[-MAX_HISTORY:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"{user_name}: {user_input}"})

    # --- 3️⃣ ATTEMPT GENERATION (With Fallback Chain) ---
    reply = await call_model_api(active_model, messages, max_tokens)

    # Fallback 1: Mistral
    if not reply and active_model != "mistral":
        reply = await call_model_api("mistral", messages, max_tokens)

    # Fallback 2: Groq
    if not reply and active_model != "groq":
        reply = await call_model_api("groq", messages, max_tokens)

    # Fallback 3: Hardcoded
    if not reply:
        return random.choice(FALLBACK_RESPONSES), is_coding_request

    # --- 4️⃣ ANTI-LOOP PROTECTION ---
    if history and len(history) >= 2:
        last_assistant = next((h['content'] for h in reversed(history) if h['role'] == 'assistant'), None)
        if last_assistant and reply.lower().strip() == last_assistant.lower().strip():
            reply = random.choice(FALLBACK_RESPONSES)

    # --- 5️⃣ SAVE MEMORY ---
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply}
    ]
    
    # Keep only recent context
    if len(new_history) > MAX_HISTORY * 2:
        new_history = new_history[-(MAX_HISTORY * 2):]
    
    chatbot_collection.update_one(
        {"chat_id": chat_id},
        {"$set": {"history": new_history}},
        upsert=True
    )

    return reply, is_coding_request


# --- 🎮 SHARED AI FUNCTION (FOR GAMES/OTHER FEATURES) ---
async def ask_mistral_raw(system_prompt, user_input, max_tokens=150):
    """Quick AI call without memory (for games, etc.)"""
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    res = await call_model_api("mistral", msgs, max_tokens)
    if not res:
        res = await call_model_api("groq", msgs, max_tokens)
    return res


# --- ⚙️ SETTINGS MENU ---

async def chatbot_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /chatbot command - Settings panel
    - PMs: Always enabled (can't disable, only switch model)
    - Groups: Admins can enable/disable + switch model
    """
    chat = update.effective_chat
    user = update.effective_user

    # Private Message: Show model switcher only
    if chat.type == ChatType.PRIVATE:
        doc = chatbot_collection.find_one({"chat_id": chat.id})
        curr_model = doc.get("model", DEFAULT_MODEL) if doc else DEFAULT_MODEL
        
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("🦙 Groq", callback_data="ai_set_groq"),
                InlineKeyboardButton("🌟 Mistral", callback_data="ai_set_mistral")
            ],
            [InlineKeyboardButton("🖥️ Codestral (Code)", callback_data="ai_set_codestral")],
            [InlineKeyboardButton("🗑️ Clear Memory", callback_data="ai_reset")]
        ])
        
        return await update.message.reply_text(
            f"🤖 <b>Baka AI Settings</b>\n\n"
            f"📍 <b>Current Model:</b> {curr_model.title()}\n"
            f"💡 <b>Tip:</b> Codestral auto-activates for code requests!",
            parse_mode=ParseMode.HTML,
            reply_markup=kb
        )

    # Group Chat: Admin check
    member = await chat.get_member(user.id)
    if member.status not in ['administrator', 'creator']:
        return await update.message.reply_text(
            "❌ Only admins can change AI settings!",
            parse_mode=ParseMode.HTML
        )

    # Get current settings
    doc = chatbot_collection.find_one({"chat_id": chat.id})
    is_enabled = doc.get("enabled", True) if doc else True
    curr_model = doc.get("model", DEFAULT_MODEL) if doc else DEFAULT_MODEL
    
    status_emoji = "🟢" if is_enabled else "🔴"
    status_text = "Enabled" if is_enabled else "Disabled"

    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Enable", callback_data="ai_enable"),
            InlineKeyboardButton("❌ Disable", callback_data="ai_disable")
        ],
        [
            InlineKeyboardButton("🦙 Groq", callback_data="ai_set_groq"),
            InlineKeyboardButton("🌟 Mistral", callback_data="ai_set_mistral")
        ],
        [InlineKeyboardButton("🖥️ Codestral (Code)", callback_data="ai_set_codestral")],
        [InlineKeyboardButton("🗑️ Clear Memory", callback_data="ai_reset")]
    ])
    
    await update.message.reply_text(
        f"🤖 <b>Baka AI Settings</b>\n\n"
        f"📊 <b>Status:</b> {status_emoji} {status_text}\n"
        f"🧠 <b>Model:</b> {curr_model.title()}\n"
        f"💡 <b>Tip:</b> Codestral auto-activates for code!",
        parse_mode=ParseMode.HTML,
        reply_markup=kb
    )


async def chatbot_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks in /chatbot menu"""
    query = update.callback_query
    data = query.data
    chat_id = query.message.chat.id
    chat_type = query.message.chat.type

    # Admin check (only for groups)
    if chat_type != ChatType.PRIVATE:
        mem = await query.message.chat.get_member(query.from_user.id)
        if mem.status not in ['administrator', 'creator']:
            return await query.answer("❌ Admin Only", show_alert=True)

    # --- ENABLE/DISABLE (Groups only) ---
    if data == "ai_enable":
        if chat_type == ChatType.PRIVATE:
            return await query.answer("⚠️ AI is always on in PMs!", show_alert=True)
        
        chatbot_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"enabled": True}},
            upsert=True
        )
        await query.answer("✅ Baka is now active!", show_alert=True)
        await query.message.edit_text(
            "✅ <b>Baka AI Enabled!</b>\n\nShe'll respond to:\n• Replies to her messages\n• @mentions\n• Messages starting with 'hey', 'hi', 'baka'",
            parse_mode=ParseMode.HTML
        )

    elif data == "ai_disable":
        if chat_type == ChatType.PRIVATE:
            return await query.answer("⚠️ Can't disable in PMs!", show_alert=True)
        
        chatbot_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"enabled": False}},
            upsert=True
        )
        await query.answer("❌ Baka is now silent!", show_alert=True)
        await query.message.edit_text(
            "🔇 <b>Baka AI Disabled</b>\n\nUse /chatbot to re-enable anytime.",
            parse_mode=ParseMode.HTML
        )

    # --- MODEL SWITCHING ---
    elif data in ["ai_set_groq", "ai_set_mistral", "ai_set_codestral"]:
        model_map = {
            "ai_set_groq": "groq",
            "ai_set_mistral": "mistral",
            "ai_set_codestral": "codestral"
        }
        new_model = model_map[data]
        
        chatbot_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"model": new_model}},
            upsert=True
        )
        
        model_names = {
            "groq": "🦙 Groq (Fast)",
            "mistral": "🌟 Mistral (Smart)",
            "codestral": "🖥️ Codestral (Code Specialist)"
        }
        
        await query.answer(f"Switched to {model_names[new_model]}!", show_alert=True)
        
        # Refresh menu with updated model
        doc = chatbot_collection.find_one({"chat_id": chat_id})
        is_enabled = doc.get("enabled", True) if doc else True
        status_emoji = "🟢" if is_enabled else "🔴"
        
        kb = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Enable", callback_data="ai_enable"),
                InlineKeyboardButton("❌ Disable", callback_data="ai_disable")
            ] if chat_type != ChatType.PRIVATE else [],
            [
                InlineKeyboardButton("🦙 Groq", callback_data="ai_set_groq"),
                InlineKeyboardButton("🌟 Mistral", callback_data="ai_set_mistral")
            ],
            [InlineKeyboardButton("🖥️ Codestral", callback_data="ai_set_codestral")],
            [InlineKeyboardButton("🗑️ Clear Memory", callback_data="ai_reset")]
        ])
        
        await query.message.edit_text(
            f"🤖 <b>Baka AI Settings</b>\n\n"
            f"{'📊 <b>Status:</b> ' + status_emoji + ('Enabled' if is_enabled else 'Disabled') if chat_type != ChatType.PRIVATE else ''}\n"
            f"🧠 <b>Model:</b> {model_names[new_model]}\n"
            f"💡 <b>Note:</b> Codestral auto-activates for code!",
            parse_mode=ParseMode.HTML,
            reply_markup=kb
        )

    # --- CLEAR MEMORY ---
    elif data == "ai_reset":
        chatbot_collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"history": []}},
            upsert=True
        )
        await query.answer("🧠 Memory wiped! Fresh start!", show_alert=True)


# --- 💬 MESSAGE HANDLER ---

async def ai_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Main handler for AI conversations
    - Always active in PMs
    - In groups: Only when enabled + (reply/mention/greeting)
    """
    msg = update.message
    if not msg:
        return
    
    chat = update.effective_chat

    # --- STICKER RESPONSE ---
    if msg.sticker:
        should_react = (
            chat.type == ChatType.PRIVATE or
            (msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id)
        )
        if should_react:
            await send_ai_sticker(update, context)
        return

    # --- TEXT PROCESSING ---
    if not msg.text or msg.text.startswith("/"):
        return
    
    text = msg.text.strip()
    if not text:
        return

    # --- DECIDE IF SHOULD REPLY ---
    should_reply = False

    if chat.type == ChatType.PRIVATE:
        # Always reply in PMs
        should_reply = True
    else:
        # Groups: Check if enabled
        doc = chatbot_collection.find_one({"chat_id": chat.id})
        is_enabled = doc.get("enabled", True) if doc else True
        
        if not is_enabled:
            return

        # Check triggers
        bot_username = context.bot.username.lower() if context.bot.username else "bot"
        
        # 1. Reply to bot's message
        if msg.reply_to_message and msg.reply_to_message.from_user.id == context.bot.id:
            should_reply = True
        
        # 2. @mention
        elif f"@{bot_username}" in text.lower():
            should_reply = True
            text = text.replace(f"@{bot_username}", "").strip()
        
        # 3. Greeting keywords
        elif any(text.lower().startswith(kw) for kw in ["hey", "hi", "hello", "sun", "oye", "baka", "ai"]):
            should_reply = True

    # --- GENERATE RESPONSE ---
    if should_reply:
        if not text:
            text = "Hi"
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        # Get user's preferred model
        doc = chatbot_collection.find_one({"chat_id": chat.id})
        pref_model = doc.get("model", DEFAULT_MODEL) if doc else DEFAULT_MODEL

        # Get AI response
        response, is_code = await get_ai_response(
            chat.id,
            text,
            msg.from_user.first_name,
            pref_model
        )

        # --- FORMAT & SEND ---
        if is_code:
            # Code: Use Markdown for proper formatting
            await msg.reply_text(response, parse_mode=ParseMode.MARKDOWN)
        else:
            # Conversation: Use stylized text (emojis, fonts)
            styled = stylize_text(response)
            await msg.reply_text(styled)

        # Random sticker (30% chance, not for code)
        if not is_code and random.random() < 0.30:
            await send_ai_sticker(update, context)


# --- 🔧 COMMAND: /ask ---

async def ask_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Direct AI query: /ask <question>
    Always uses default model (Mistral) unless code detected
    """
    msg = update.message
    
    if not context.args:
        return await msg.reply_text(
            "💬 <b>Usage:</b> <code>/ask Your question here</code>\n\n"
            "Example: <code>/ask What's the weather like?</code>",
            parse_mode=ParseMode.HTML
        )
    
    await context.bot.send_chat_action(chat_id=msg.chat.id, action=ChatAction.TYPING)
    
    query = " ".join(context.args)
    response, is_code = await get_ai_response(
        msg.chat.id,
        query,
        msg.from_user.first_name,
        DEFAULT_MODEL
    )

    if is_code:
        await msg.reply_text(response, parse_mode=ParseMode.MARKDOWN)
    else:
        await msg.reply_text(stylize_text(response))