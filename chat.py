"""Initialize the chatbot backend."""
import fitz  # imports the pymupdf library
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.schema import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)


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
    if embedding_model is None or embedding_model == "openai":
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
        llm = ChatOpenAI(metadata={"name": "question_generator"})
        streaming_llm = ChatOpenAI(
            streaming=True, metadata={"name": "answer_generator"}
        )
    elif llm_model.startswith("openai:"):
        model_name = llm_model.replace("openai:", "")
        llm = ChatOpenAI(metadata={"name": "question_generator"})
        streaming_llm = ChatOpenAI(
            model_name=model_name,
            streaming=True,
            metadata={"name": "answer_generator"},
        )
    elif llm_model.startswith("huggingface:"):
        repo_id = "google/flan-t5-xxl"
        model_kwargs = {
            "temperature": 0.5,
            "max_length": 512,
        }
        llm = HuggingFaceHub(
            repo_id=repo_id,
            metadata={"name": "question_generator"},
            model_kwargs=model_kwargs,
        )
        streaming_llm = HuggingFaceHub(
            repo_id=repo_id,
            metadata={"name": "answer_generator"},
            model_kwargs=model_kwargs
            | {
                "stream": True,
            },
        )
    question_generator = LLMChain(
        llm=llm, prompt=CONDENSE_QUESTION_PROMPT,
    )
    # Streaming doc chain to combine the reponses
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=QA_PROMPT,
    )

    # Memory to store the chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        tags=["constructed_chain"],
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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
        self.llm_name = ""

    def on_llm_start(self, serialized, inputs, **kwargs):
        self.llm_name = kwargs["metadata"]["name"]
        if self.llm_name != "answer_generator":
            self.container.markdown("Thinking ...")

    def on_llm_new_token(self, token: str, **kwargs):
        # Only stream the answer generator
        if self.llm_name == "answer_generator":
            self.text += token
            self.container.markdown(self.text + "▌")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Run when LLM ends running."""
        self.container.markdown(self.text)
