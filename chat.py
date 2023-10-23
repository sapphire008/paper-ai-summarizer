"""Initialize the chatbot backend."""
import fitz  # imports the pymupdf library
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        with fitz.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:  # iterate the document pages
                text += page.get_text()  #  get plain text encoded as UTF-8
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, embedding_model=None):
    if embedding_model is None:
        embeddings = OpenAIEmbeddings()
    elif embedding_model.startswith("huggingface"):
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=embedding_model.replace("huggingface:", "")
            or "hkunlp/instructor-xl"
        )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, llm_model=None):
    if llm_model is None:
        # gpt-3.5-turbo by default
        llm = ChatOpenAI()
    elif llm_model.startswith("openai:"):
        llm = ChatOpenAI(llm_model.replace("openai:", ""))
    elif llm_model.startswith("huggingface:"):
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def initialize_conversation(pdf_docs, settings={}):
    # get pdf text
    raw_text = get_pdf_text(pdf_docs)

    # get the text chunks
    text_chunks = get_text_chunks(raw_text)

    # create vector store
    vectorstore = get_vectorstore(
        text_chunks, embedding_model=settings.get("EMBEDDING_MODEL")
    )

    # create conversation chain
    return get_conversation_chain(
        vectorstore, llm_model=settings.get("LLM_MODEL")
    )
