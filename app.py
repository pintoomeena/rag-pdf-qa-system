import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with your PDF")
st.caption("Upload a document and ask questions about its contents.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Sidebar: PDF upload
with st.sidebar:
    st.header("📄 Upload Documents")

    pdf_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload and Process", use_container_width=True):
        if not pdf_files:
            st.warning("Please select at least one PDF.")
        elif len(pdf_files) > 20:
            st.error("You can upload a maximum of 20 PDFs.")
        else:
            try:
                files = [
                    (
                        "files",
                        (
                            pdf.name,
                            pdf.getvalue(),
                            "application/pdf"
                        )
                    )
                    for pdf in pdf_files
                ]

                with st.spinner("Processing your PDFs..."):
                    response = requests.post(
                        f"{API_URL}/upload/",
                        files=files,
                        timeout=600
                    )

                if response.ok:
                    result = response.json()

                    if result.get("error"):
                        st.error(result["error"])
                    else:
                        st.session_state.uploaded_files = (
                            result.get("files", [])
                        )
                        st.session_state.messages = []
                        st.success("PDFs processed successfully!")
                else:
                    st.error(
                        f"Upload failed: {response.status_code} "
                        f"{response.text}"
                    )

            except requests.RequestException as e:
                st.error(f"Could not connect to the backend: {e}")

    if st.session_state.uploaded_files:
        st.subheader("Uploaded PDFs")

        for filename in st.session_state.uploaded_files:
            st.write(f"📄 {filename}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Relevant document passages"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Passage {i}**")
                    st.write(source)

# Chat input
question = st.chat_input("Ask a question about your PDF...")

if question:
    if not st.session_state.uploaded_files:
        st.warning("Please upload and process a PDF first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents..."):
                    response = requests.get(
                        f"{API_URL}/query/",
                        params={"question": question},
                        timeout=300
                    )

                response.raise_for_status()
                result = response.json()

                answer = result.get(
                    "answer",
                    "No answer was returned."
                )

                sources = result.get("relevant_chunks", [])

                st.markdown(answer)

                if sources:
                    with st.expander("Relevant document passages"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Passage {i}**")
                            st.write(source)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except requests.RequestException as e:
                error_message = (
                    f"Unable to get an answer from the backend: {e}"
                )
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })