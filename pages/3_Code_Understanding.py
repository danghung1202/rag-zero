
import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import WebBaseLoader
import langchain_rag as lrag
import file_utils as fu
import url_utils as uu
from load_config import getOpenAIKey

from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language


OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Sidebar contents
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"

    add_vertical_space(5)
    st.write('Made with love AI')

def git_clone(url):
    repo_path = f"/Users/hung.dang/Desktop/{uu.get_last_segment(url)}"
    repo = Repo.clone_from(url, to_path=repo_path)
    return repo_path

def code_loader(url):
    repo_path = git_clone(url)
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py"],
        exclude=["**/non-utf8-encoding.py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    return loader


def main():
    st.header("Chat with your code")
    url = st.text_input("Input the github repo")
    if url is not None and url.strip():
        store_name = f"faiss_stores/{fu.create_file_name(url)}"
        embeddings = lrag.create_embeddings(OPENAI_API_KEY)

        if os.path.exists(f"{store_name}/index.pkl"):
            vectordb = lrag.load_vector_database(store_name, embeddings)
        else:
            with st.spinner("Loading and processing your code ..."):
                # Load and convert the pdf file to text
                loader = code_loader(url)
                docs = loader.load()
                if docs is not None:

                    chunks = lrag.split_code_into_chunks(docs, Language.PYTHON, CHUNK_SIZE, CHUNK_OVERLAP)
                    vectordb = lrag.save_document_embedding_to_db(chunks, embeddings, store_name)

                    # create the text file contains the web content
                    fu.create_new_file(f"{store_name}/{uu.convert_url_to_file_name(url)}.txt", f"{lrag.format_docs(docs).encode('utf-8')}")
                    st.write(f"The code is embedded to {store_name}.pkl successfully")
                else:
                    st.warning("Load and convert the code to text failed")

        question = st.text_input("Ask something about your code:")

        if (vectordb is not None) and (question.strip()):
            # docs = vectordb.similarity_search(question)
            # st.write(f"Context: {lrag.format_docs(docs)}")

            retriever = lrag.create_retriever(vectordb)
            prompt = lrag.create_prompt_template()
            model = lrag.create_model(OPENAI_API_KEY)
            output_parser = StrOutputParser()
            # create the chain
            chain = (
                # run RunnableParallel in short
                {"context": retriever | lrag.format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | output_parser
            )

            response = chain.invoke(question)
            # Another clear way to put the relevant result to context

            # chain = prompt | model | output_parser
            # response = chain.invoke({"question": question, "context": docs})

            st.write(f"Q: {question}")
            st.write(f"A: {response}")


if __name__ == '__main__':
    main()
