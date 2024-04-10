import math

import faiss
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS


def relevance_score_fn(score: float) -> float:
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1024
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=10
    )
