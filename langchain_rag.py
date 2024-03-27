from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI

# Constants and API Keys
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your actual API key
GPT_MODEL_NAME = 'gpt-4'

# Load and convert the pdf file to text
def load_and_convert_pdf_to_text(pdf):
    """Load and extract the text from pdf using the PdfReader"""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text
    return None

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Because of limitation of context window,
     need to splits text into smaller chunks for processing."""
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_spliter.split_text(text)

def create_embeddings(api_key):
    """Creates embeddings from text."""
    # Set chunk_site = 300 to take over the issue rate limit when the token larger than context window
    # https://github.com/langchain-ai/langchain/discussions/9560
    return OpenAIEmbeddings(openai_api_key=api_key, chunk_size=300)


def load_vector_database(store_name, embeddings):
    vectordb = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
    return vectordb


def save_text_embedding_to_db(chunks, embeddings, store_name):
    """Sets up a vector database for storing embeddings."""
    vectordb = FAISS.from_texts(chunks, embedding=embeddings)
    vectordb.save_local(store_name)
    return vectordb


def create_retriever(vectordb):
    retriever = vectordb.as_retriever()
    return retriever


def create_prompt_template():
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def create_model(api_key):
    model = ChatOpenAI(api_key=api_key, temperature=0)
    return model


def format_docs(docs):
    """Extract the `page_content` field from the `doc` object and join the list the docs to one string"""
    return "\n\n".join(doc.page_content for doc in docs)
