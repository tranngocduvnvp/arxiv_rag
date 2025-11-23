from langchain_community.vectorstores import Chroma
from google import genai
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import pandas as pd
import os
import ast
import string
import re
import asyncio


BATCH_EMBEDDING = 10
_MAX_TOKENS_PER_BATCH = 20000
_DEFAULT_BATCH_SIZE = 2

# ====== Tạo class Embedding cho Gemini ======
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "models/embedding-001"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
    
    @staticmethod
    def _split_by_punctuation(text: str) -> list[str]:
        """Splits a string by punctuation and whitespace characters."""
        split_by = string.punctuation + "\t\n "
        pattern = f"([{split_by}])"
        # Using re.split to split the text based on the pattern
        return [segment for segment in re.split(pattern, text) if segment]
    
    @staticmethod
    def _prepare_batches(texts: list[str], batch_size: int) -> list[list[str]]:
        """Splits texts in batches based on current maximum batch size and maximum
        tokens per request.
        """
        text_index = 0
        texts_len = len(texts)
        batch_token_len = 0
        batches: list[list[str]] = []
        current_batch: list[str] = []
        if texts_len == 0:
            return []
        while text_index < texts_len:
            current_text = texts[text_index]
            # Number of tokens per a text is conservatively estimated
            # as 2 times number of words, punctuation and whitespace characters.
            # Using `count_tokens` API will make batching too expensive.
            # Utilizing a tokenizer, would add a dependency that would not
            # necessarily be reused by the application using this class.
            current_text_token_cnt = (
                len(GeminiEmbeddings._split_by_punctuation(current_text))
                * 2
            )
            end_of_batch = False
            if current_text_token_cnt > _MAX_TOKENS_PER_BATCH:
                # Current text is too big even for a single batch.
                # Such request will fail, but we still make a batch
                # so that the app can get the error from the API.
                if len(current_batch) > 0:
                    # Adding current batch if not empty.
                    batches.append(current_batch)
                current_batch = [current_text]
                text_index += 1
                end_of_batch = True
            elif (
                batch_token_len + current_text_token_cnt > _MAX_TOKENS_PER_BATCH
                or len(current_batch) == batch_size
            ):
                end_of_batch = True
            else:
                if text_index == texts_len - 1:
                    # Last element - even though the batch may be not big,
                    # we still need to make it.
                    end_of_batch = True
                batch_token_len += current_text_token_cnt
                current_batch.append(current_text)
                text_index += 1
            if end_of_batch:
                batches.append(current_batch)
                current_batch = []
                batch_token_len = 0
        return batches

    async def _embed_documents_async(self, texts):
        embeddings = []
        batches = GeminiEmbeddings._prepare_batches(texts, _DEFAULT_BATCH_SIZE)
        # print(batches)
        
        tasks = [
            self.client.aio.models.embed_content(
                model=self.model,
                contents=batch,
            )
            for batch in batches
        ]


        # chạy song song
        results = await asyncio.gather(*tasks)
        # print(results)

        # gom kết quả lại
        embeddings = []
        for res in results:
            embeddings.extend([list(e.values) for e in res.embeddings])

        return embeddings
     
    def embed_documents(self, texts):
        return asyncio.run(self._embed_documents_async(texts))

    def embed_query(self, text):
        emb = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )
        print(emb)
        emb = list(emb.embeddings[0].values)
        return emb

# ====================== MAIN =======================

def build_chroma_from_csv(path_csv, persist_dir, collection_name, gemini_api_key):

    # Dùng Gemini thay cho HuggingFace
    gemini_emb = GeminiEmbeddings(
        api_key=gemini_api_key,
        model="gemini-embedding-001"
    )

    # Đọc dữ liệu CSV
    chunks = []
    df = pd.read_csv(path_csv)

    for i in range(len(df)):
        row = df.iloc[i]
        chunks.append(
            Document(
                page_content=row["page_content"],
                metadata=ast.literal_eval(row["metadata"])
            )
        )
        if i==1:
            break

    # Xóa ChromaDB cũ nếu có
    import shutil
    chroma_path = os.path.join(persist_dir, collection_name)
    if os.path.exists(chroma_path):
        print(f"Delete ChromaDB: {chroma_path}")
        shutil.rmtree(chroma_path)

    # Tạo VectorDB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=gemini_emb,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    vectordb.persist()
    return vectordb

path_csv = "/home/dutn/Documents/ARXIVRAG/RAG_VT/output/save_chunk.csv"
persist_dir = "/home/dutn/Documents/ARXIVRAG/RAG_VT/output" 
collection_name = "gemini_db"
gemini_api_key = "AIzaSyCwDT5DABNsSAJCdGUyjktUv3oo-C-FgHk"

vectordb = build_chroma_from_csv(path_csv, persist_dir, collection_name, gemini_api_key)

vectorstore_retreiver = vectordb.as_retriever(search_kwargs={"k": 2})
result_retrievled = vectorstore_retreiver.invoke("xin chao")
print(result_retrievled)