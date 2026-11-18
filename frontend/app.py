import streamlit as st
import requests

# App custom page config
st.set_page_config(
    page_title="Intelligent Web-RAG Agent",
    page_icon="🤖",
    layout="wide"
)

# Configuration backend API url
API_BASE_URL = "http://127.0.0.1:8000"

st.title("🤖 Intelligent Web-RAG Agent")
st.caption("Enter a URL to ingest, and chat with its live contents instantly.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    target_url = st.text_input("Enter Webpage URL:", placeholder="https://en.wikipedia.org/wiki/Coimbatore")
    
    if st.button("📥 Ingest & Vectorize", use_container_width=True):
        if not target_url:
            st.error("Please provide a valid URL first!")
        else:
            with st.spinner("Ingesting webpage... (Executing Selenium/BS4 fallback)"):
                try:
                    response = requests.post(f"{API_BASE_URL}/ingest", json={"url": target_url})
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Successfully processed webpage!")
                        st.balloons()
                        # Track URL & sub-pages in state
                        st.session_state.current_url = target_url
                        st.session_state.char_count = data.get("character_count", 0)
                        st.session_state.ingested_sub_pages = data.get("pages", []) # Catch them
                    else:
                        st.error(f"Error ({response.status_code}): {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Failed to reach Backend: {e}")

    st.divider()
    if 'current_url' in st.session_state:
        st.write(f"🟢 **Active URL:** {st.session_state.current_url}")
        st.write(f"📄 **Size:** {st.session_state.char_count} characters loaded")
        
        # Show exactly what pages were ingested
        if st.session_state.get("ingested_sub_pages"):
             st.write("📂 **Vectorized Pages (boundaries):**")
             for page in st.session_state.ingested_sub_pages:
                  st.caption(f"- {page}")
    else:
        st.write("🔴 No active webpage ingested.")

# Initialize Chat History for User Context
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show current Chat UI conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Chat Input
if user_query := st.chat_input("Ask any question about the ingested page..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Render AI streaming response placeholder
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document context..."):
            try:
                # Query our FastAPI server
                response = requests.post(
                    f"{API_BASE_URL}/query", 
                    json={"question": user_query}
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    source = data.get("source")
                    
                    st.markdown(answer)
                    st.caption(f"Source Reference: [{source}]({source})")
                    st.session_state.messages.append({"role": "assistant", "content": f"{answer}\n\n*Ref: {source}*"})
                else:
                    err_msg = response.json().get("detail", "Failed query validation.")
                    st.error(f"⚠️ Error: {err_msg}")
            except Exception as e:
                st.error(f"Network error communicating with API backend: {e}")