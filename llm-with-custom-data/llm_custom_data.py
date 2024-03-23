import streamlit as st

st.set_page_config(page_title="Upload PDFs and chat with their content",
                   page_icon=":books:")
st.header("Upload PDFs and chat with their content :books:")
st.subheader("Type your question below")
with st.sidebar:
    st.subheader("Data collection center")
    pdf_docs = st.file_uploader(
        "Upload your PDF files here", accept_multiple_files=True)
st.text_input(label="", placeholder="Start chatting")
st.caption('Response:')

