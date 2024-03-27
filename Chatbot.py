

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain_core.output_parsers import StrOutputParser
import langchain_rag as lrag

from load_config import getOpenAIKey

OPENAI_API_KEY = getOpenAIKey()
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Sidebar contents
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/danghung1202/rag-zero)"

    add_vertical_space(5)
    st.write('Made with love AI')


def main():
    st.header("Chatbot")
    template = """You are world class technical documentation write.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always say "thanks for asking!" at the end of the answer.

    Question: {question}

    Helpful Answer:"""
    prompt = lrag.create_prompt_template(template)
    model = lrag.create_model(OPENAI_API_KEY)
    output_parser = StrOutputParser()
    # create the chain
    chain = prompt | model | output_parser

    
    question = st.text_input("Enter your question:")
    if question.strip():
        response = chain.invoke(question)
        st.write(f"Q: {question}")
        st.write(f"A: {response}")


if __name__ == '__main__':
    main()
