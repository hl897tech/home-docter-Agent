import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
logger = logging.getLogger(__name__)

_vectorstore = None

KB_PATH = Path(__file__).parent / "knowledge_base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def _build_vectorstore() -> FAISS:
    loader = DirectoryLoader(
        str(KB_PATH),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    docs = loader.load()
    if not docs:
        raise ValueError(f"No documents found in {KB_PATH}")
    logger.info("Loaded %d documents from knowledge base", len(docs))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks", len(chunks))

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return FAISS.from_documents(chunks, embeddings)


def get_retriever(k: int = 3):
    """Return a LangChain retriever backed by the FAISS vectorstore (lazy init)."""
    global _vectorstore
    if _vectorstore is None:
        logger.info("Building FAISS vectorstore...")
        _vectorstore = _build_vectorstore()
        logger.info("FAISS vectorstore ready")
    return _vectorstore.as_retriever(search_kwargs={"k": k})