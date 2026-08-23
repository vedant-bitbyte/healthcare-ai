"""Streamlit UI for AI Sarthi."""

import logging
# import os
# import sys
import traceback

# Add project root to PYTHONPATH so we can import app and src modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from app.chat_service import get_document_count, get_rag_pipeline

# Configure basic logging for the app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Sarthi - Healthcare AI",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource(show_spinner="Initializing RAG Pipeline (Loading model & database)...")
def load_pipeline():
    """Cache the pipeline so it's not recreated on every interaction."""
    return get_rag_pipeline()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []


def main():
    init_session_state()

    # Load pipeline
    try:
        pipeline = load_pipeline()
    except Exception as exc:
        st.error("Failed to load RAG Pipeline. Please check the application logs.")
        logger.error("Pipeline loading error: %s\n%s", exc, traceback.format_exc())
        return

    # Sidebar
    with st.sidebar:
        st.title("🏥 AI Sarthi")
        st.subheader("Healthcare AI Assistant")
        st.divider()

        st.markdown("**Model:** `Fine-Tuned Phi-3 Mini`")
        st.markdown("**Retrieval:** `ChromaDB`")

        doc_count = get_document_count(pipeline)
        st.markdown(f"**Documents Indexed:** `{doc_count}`")

        st.divider()
        st.button("New Chat", on_click=clear_chat, use_container_width=True, type="primary")
        st.divider()

        st.info(
            "**Disclaimer:** This AI assistant provides informational responses "
            "based on retrieved healthcare documents and should not replace "
            "professional medical advice."
        )

    # Main Chat Area
    st.title("AI Sarthi")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Sources"):
                    for source in msg["sources"]:
                        st.markdown(f"📄 `{source}`")

    # Chat Input
    if prompt := st.chat_input("Ask a healthcare question..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add to session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching healthcare documents and generating response..."):
                    result = pipeline.run(prompt)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            st.markdown(f"📄 `{source}`")

                # Add to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_msg = "Sorry, I encountered an error while processing your request."
                st.error(f"{error_msg} (See logs for details)")
                logger.error("Generation error: %s\n%s", exc, traceback.format_exc())
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    }
                )


if __name__ == "__main__":
    main()
