import streamlit as st
import time

st.set_page_config(page_title="Chat History", layout="wide")

st.title("📜 Chat History")
st.caption("All Q&A pairs from the current session")

# ─── CHECK SESSION STATE ─────────────────────────────────────────────────────
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.info("No chat history yet. Go to the **💬 Chat** page, upload a PDF, and ask some questions!")
    st.stop()

# ─── STATS ───────────────────────────────────────────────────────────────────
user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]

col1, col2, col3 = st.columns(3)
col1.metric("Total Messages", len(st.session_state.messages))
col2.metric("Your Questions", len(user_msgs))
col3.metric("AI Responses", len(assistant_msgs))

st.divider()

# ─── DISPLAY ALL Q&A PAIRS ──────────────────────────────────────────────────
pair_num = 0
for i in range(0, len(st.session_state.messages), 2):
    pair_num += 1
    user_msg = st.session_state.messages[i] if i < len(st.session_state.messages) else None
    ai_msg = st.session_state.messages[i + 1] if (i + 1) < len(st.session_state.messages) else None

    with st.expander(f"💬 Exchange {pair_num}: {user_msg['content'][:80]}..." if user_msg else f"Exchange {pair_num}", expanded=(pair_num <= 3)):

        if user_msg:
            st.markdown(f"**🧑 You:**")
            st.info(user_msg["content"])

        if ai_msg:
            st.markdown(f"**🤖 AI:**")
            st.success(ai_msg["content"])
        else:
            st.warning("⏳ Waiting for AI response...")

st.divider()

# ─── EXPORT OPTION ───────────────────────────────────────────────────────────
st.subheader("📥 Export History")

export_text = ""
for i, msg in enumerate(st.session_state.messages):
    role = "You" if msg["role"] == "user" else "AI"
    export_text += f"[{role}]: {msg['content']}\n\n"

st.download_button(
    label="📄 Download as Text",
    data=export_text,
    file_name=f"chat_history_{time.strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)
