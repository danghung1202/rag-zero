

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.docstore.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.llm import LLMChain
import langchain_rag as lrag

from load_config import getOpenAIKey

OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
template = """Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:"""


def create_summarize_chain(model, prompt):
    llm_chain = LLMChain(llm=model, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
    return stuff_chain

# this function is responsible for splitting the data into smaller chunks
# and convert the data in document format


def chunks_and_document(pdf):
    text = lrag.load_and_convert_pdf_to_text(pdf)
    chunks = lrag.split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    docs = [Document(page_content=t) for t in chunks]  # convert the splitted chunks into document format

    return docs


# Sidebar contents
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/danghung1202/rag-zero.git)"

    add_vertical_space(5)
    st.write('Made with love AI')


def main():
    st.header("Document Summarization")

    prompt = lrag.create_prompt_template(template)
    model = lrag.create_model(OPENAI_API_KEY)
    # create the chain
    summarize_chain = create_summarize_chain(model, prompt)

    use_document = "Upload a document"
    use_url = "Enter a website URL"
    input_method = st.radio("Select source knowledge", (use_document, use_url))

    if input_method == use_document:
        # upload a PDF file
        pdf = st.file_uploader("Upload your document", type='pdf')
    
    elif input_method == use_url:
        # extract the content from url
        url = st.text_input("Input the url you want to summary")
        
    if st.button('Summarize'):
        if input_method == use_document:
            if pdf is not None:
                with st.spinner("Summarizing the pdf..."):
                    # convert the splitted chunks into document format
                    docs = chunks_and_document(pdf)
                    response = summarize_chain.run(docs)
                    st.write(f"Summarized: {response}")

        elif input_method == use_url:
            if url is not None and url.strip():
                with st.spinner("Summarizing the url..."):
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    response = summarize_chain.run(docs)
                    st.write(f"Summarized: {response}")
    
if __name__ == '__main__':
    main()
