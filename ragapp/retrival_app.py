import warnings
warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Tuple
from pydantic import Field
from langchain.retrievers import  EnsembleRetriever
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from rerankmd import ranking
import warnings
warnings.filterwarnings("ignore")



# --- Khởi tạo ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
PERSIST_DIR = "/data/AIRACE/arxiv_rag/output"
CHROMA_COLLECTION_NAME = "arxiv_chunk"
GEMINI_API_KEY = "AIzaSyCwDT5DABNsSAJCdGUyjktUv3oo-C-FgHk"


embedding_fn = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=CHROMA_COLLECTION_NAME,
    embedding_function=embedding_fn,
)

data = vectorstore.get()
docs = [Document(page_content=d, metadata=m) for d, m in zip(data["documents"], data["metadatas"])]
bm25_retriever = BM25Retriever.from_documents(docs)


def retriever_hybrid(vectorstore, bm25_retriever, k, vector_score, metadata_filter):
    bm25_retriever.k = k
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": k, "filter": metadata_filter})
    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                    bm25_retriever],
                                        weights=[vector_score, 1- vector_score])
    return ensemble_retriever


def fuse_retrival_rerank(query, model, tokenizer, vectorstore, \
                         bm25_retriever, k, vector_score=0.99, \
                         is_ranking=False, metadata_filter=None):
    ensemble_retriever = retriever_hybrid(vectorstore, bm25_retriever, k, vector_score, metadata_filter)
    result_retrievled = ensemble_retriever.invoke(query)
    documents = [doc.page_content for doc in result_retrievled]
    if is_ranking==True:
        inputs, rank_score = ranking(query, documents, model, tokenizer)
        fused_docs = [{"document": doc.page_content, "metadata": doc.metadata, "rerank_score": float(score)} for doc, score in zip(result_retrievled, rank_score)]
        reranked_docs = sorted(fused_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs
    else:
        return [{"document": doc.page_content, "metadata": doc.metadata} for doc in result_retrievled]

if __name__  == "__main__":

    question = r"what is the accuracy of yolov8?"
    reranked_docs = fuse_retrival_rerank(question, None, None, vectorstore, bm25_retriever, \
                                         5, 0.99999, is_ranking=False, metadata_filter={'Header 1': 'REFERENCES'})

    #sắp xếp theo rankscore
    for i, doc in enumerate(reranked_docs, 1):
        # print(doc)
        print(f"\n------------- Kết quả {i}-----------------")
        print(doc["metadata"])
        print(doc["document"])
