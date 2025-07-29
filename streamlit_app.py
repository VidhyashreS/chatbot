import os
import concurrent.futures
from dotenv import load_dotenv

import streamlit as st
from groq import Groq
from googletrans import Translator

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("Loaded GROQ_API_KEY:", GROQ_API_KEY)


@st.cache_resource(show_spinner=False)
def get_model():
    """Initialise the Groq client once per session/container."""
    return Groq(api_key=GROQ_API_KEY)


model = get_model()
translator = Translator()

SYSTEM_PROMPT = (
    "You are an expert on Andhra Pradesh tourism and cuisine. "
    "Whenever asked about food, provide detailed information about famous dishes, "
    "street food, regional specialties, and culinary traditions from all parts "
    "of Andhra Pradesh."
)

LANG_MAP = {"English": "en", "Hindi": "hi", "Telugu": "te"}

st.set_page_config(page_title="Saanchari – AP Tourism Chatbot", layout="wide")

# Inject CSS for compact language selector
st.markdown(
    """
    <style>
        .lang-select {
            position:absolute; top:18px; left:24px;
            background:#fff; border-radius:6px;
            box-shadow:0 1px 3px rgba(0,0,0,0.08);
            padding:0.05rem 0.35rem; font-size:0.85rem; z-index:1000;
        }
        .lang-select label {display:none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Language picker (non-empty label prevents Streamlit warning)
st.markdown("<div class='lang-select'>", unsafe_allow_html=True)
selected_lang = st.selectbox(
    label="Language",                 # must be non-empty
    options=list(LANG_MAP.keys()),
    index=0,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

# Header
st.markdown(
    "<h1 style='color:#07546B; margin-top:0;'>Saanchari – Andhra Pradesh Tourism Chatbot</h1>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

BUILTIN_QUESTIONS = [
    "What are the top tourist attractions in Andhra Pradesh?",
    "Tell me about the famous food in Andhra Pradesh.",
    "What is the best time to visit Andhra Pradesh?",
]

col1, col2, col3 = st.columns(3)
for col, text in zip((col1, col2, col3), BUILTIN_QUESTIONS):
    if col.button(text, use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": text})


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask something about Andhra Pradesh Tourism..."):
    st.session_state.messages.append({"role": "user", "content": prompt})


def fetch_response(user_prompt: str) -> str:
    """Call Groq with timeout and return plain text (or error)."""
    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                lambda: model.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=1,
                    max_completion_tokens=1024,
                    top_p=1,
                    stream=False,
                    stop=None,
                )
            )
            result = future.result(timeout=20)
        return result.choices[0].message.content.strip()
    except Exception as exc:
        return f"⚠️ Groq API Error: {exc}"


if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_msg = st.session_state.messages[-1]["content"]
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_msg}"

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = fetch_response(full_prompt)

            # Translate if needed
            target_lang = LANG_MAP[selected_lang]
            if target_lang != "en":
                try:
                    reply = translator.translate(reply, dest=target_lang).text
                except Exception as exc:
                    reply = f"⚠️ Translation Error: {exc}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})


st.markdown(
    """
    <div style='position:fixed; bottom:0; left:0; right:0; background:#FFF;
                border-top:1px solid #CFD1D1; padding:0.5rem; text-align:center;'>
        <small style='color:#07546B;'>© 2025 Kshipani Tech Ventures Pvt Ltd. All rights reserved.</small>
    </div>
    """,
    unsafe_allow_html=True,
)

