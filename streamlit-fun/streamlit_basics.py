import streamlit as st

st.set_page_config(page_title="This is a page",
                   page_icon=":car:")
st.header("This is flying header :airplane:")
st.subheader("This is flying sub-header")
st.text_input("Okay, here is a textbox")
st.balloons()
with st.sidebar:
    st.subheader("Left sidebar")
    pdf_docs = st.file_uploader(
        "This is a file uploader", accept_multiple_files=True)
