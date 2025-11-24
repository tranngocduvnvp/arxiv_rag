from chonkie import Pipeline, FileFetcher
from chonkie import RecursiveRules, RecursiveLevel, TableChunker, CodeChunker
from transformers import AutoTokenizer, AutoModelForCausalLM
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
import os
from typing import List, Optional, Dict
import glob
from pathlib import Path
from chonkie import MarkdownChef, RecursiveChunker, OverlapRefinery
import copy
from transformers import pipeline
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from gemini_emb import GeminiEmbeddings
import warnings
warnings.filterwarnings("ignore")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
PERSIST_DIR = "/data/AIRACE/arxiv_rag/output"
CHROMA_COLLECTION_NAME = "arxiv_chunk"
GEMINI_API_KEY = "AIzaSyBybSMJbI0oaKQLPUsYK1CBflIfzRwT0HU"


class MarkdownPasser(MarkdownChef):
    def __init__(self, tokenizer = "character"):
        super().__init__(tokenizer)
        self.chunker_header = MarkdownHeaderTextSplitter(
            [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
            ], strip_headers=True
        )
    
    def process(self, path):
        # Read the markdown file
        markdown = self.read(path)
        md_header_splits = self.chunker_header.split_text(markdown) #list[Document(page_content='Hi this is Jim  \nHi this is Joe', metadata={'Header 1': 'Foo'})]
        
        chunk_respect_to_header = []
        for md_hd in md_header_splits:
            page_content = md_hd.page_content
            meta_data = md_hd.metadata
            chunk_respect_to_header.append((self.parse(page_content), meta_data))
        
        return chunk_respect_to_header

def chat_with_llm(messages, base_url="http://0.0.0.0:23333/v1"):
    # client = OpenAI(base_url=base_url, api_key="EMPTY")
    client = OpenAI(api_key=GEMINI_API_KEY,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    completion = client.chat.completions.create(
        model="gemini-2.5-flash-lite",
        messages=messages
    )
    return completion.choices[0].message.content



class DocumentProcesser:
    def __init__(self, dir=None, path=None, chunk_size=512, overlap_size=0.3, tokenizer_chunk=None,\
                    use_summarize_code=False, use_summarize_table=False,\
                    split_table=False, table_size=2048, rules=None, distance= 100, chat_with_llm=None, \
                    detect_language=False, append_text_header=True, append_table_header=True, append_code_header=True):

        if path is not None:
            self.list_file = [path]
        else:
            self.list_file = glob.glob(dir + "/*.md")

        self.markdown_passer = MarkdownPasser()

        overlap_size = int(chunk_size*overlap_size)
        chunk_size = chunk_size - overlap_size

        if rules == None:
            rules = RecursiveRules(
                                    levels=[
                                        RecursiveLevel(delimiters=["\n\n"], include_delim="next"),
                                        RecursiveLevel(delimiters=["\n"], include_delim="next"),
                                        RecursiveLevel(delimiters=["."], include_delim="next"),
                                        RecursiveLevel(delimiters=[","], include_delim="next"),
                                        RecursiveLevel(whitespace=False)
                                    ]
                            )

        self.tokenizer_chunk = AutoTokenizer.from_pretrained(tokenizer_chunk) \
            if tokenizer_chunk is not None else "character"

        self.chunker_table = TableChunker(
            tokenizer=self.tokenizer_chunk,  # Default tokenizer (or use "gpt2", etc.)
            chunk_size=table_size         # Maximum tokens or characters per chunk
        )

        self.chunker_text = RecursiveChunker(
            tokenizer=self.tokenizer_chunk,
            chunk_size=chunk_size,
            min_characters_per_chunk = 1
        )
        
        self.overlap_refinery = OverlapRefinery(
            tokenizer=self.tokenizer_chunk,
            context_size=overlap_size,
            method="suffix",
            merge=True,
            rules=rules,
            mode="recursive"
        )

        if chat_with_llm is not None:
            # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", cache_dir = "/data/AIRACE/RAG")
            # model_llm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", cache_dir = "/data/AIRACE/RAG")
            # self.pipe = pipeline("text-generation", model=model_llm, tokenizer=tokenizer)
            self.chat_with_llm = chat_with_llm


        self.split_table = split_table
        self.distance = distance
        self.use_summarize_code = use_summarize_code
        self.use_summarize_table = use_summarize_table
        self.detect_language = detect_language
        self.table_size = table_size
        self.append_text_header = append_text_header
        self.append_table_header = append_table_header
        self.append_code_header = append_code_header
        self.chunkings = []
        
    def process(self):
        
        for path in self.list_file:
            chunk_respect_to_header = self.markdown_passer.process(path)
            for chunk_header in chunk_respect_to_header:
                md_content, metadata = chunk_header
                metadata.update({"document_file": path})
                text_chunk = md_content.chunks
                table_chunk = md_content.tables
                code_chunk = md_content.code
                content = md_content.content
                image_chunk = md_content.images

                text_smaller_chunks = self.text_chunking(text_chunk, metadata)
                self.chunkings +=text_smaller_chunks
                code_small_chunks = self.code_chunking(code_chunk, metadata)
                self.chunkings +=code_small_chunks
                table_small_chunks = self.table_chunking(table_chunk, metadata)
                self.chunkings +=table_small_chunks

        return self.chunkings

    def text_chunking(self, content, metadata):
        '''
            chunking text into small chunk
        '''
        text_chunks = []
        #chunk text
        for text in content:
            chunks = self.chunker_text.chunk(text.text)
            chunks = self.overlap_refinery(chunks)
            for ct in chunks:
                updated_metadata = copy.deepcopy(metadata)
                updated_metadata.update({"type":"text"})

                if self.append_text_header:
                    embedding_text = self.prepare_for_embedding(updated_metadata, ct.text, component_type="text")
                else:
                    embedding_text = ct.text
                text_chunks.append(Document(page_content=embedding_text, metadata=updated_metadata))
        return text_chunks
    
    def code_chunking(self, content, metadata):
        code_small_chunks = []
        for code in content:
            code_content = code.content
            if self.detect_language:
                #check code language using llm
                language = "c"
            else:
                language = "auto"

            chunker = CodeChunker(
                language=language,      # Specify the programming language
                tokenizer="character",  # Default tokenizer (or use "gpt2", etc.)
                chunk_size=2048,        # Maximum tokens per chunk
                include_nodes=False     # Optionally include AST nodes in output
            )
            # print(code_content)
            code_chunks = chunker(code_content)
            for cchunk in code_chunks:
                updated_metadata = copy.deepcopy(metadata)
                updated_metadata.update({"type": "code"})
                if not self.use_summarize_code:
                    if self.append_code_header:
                        sm_code = self.prepare_for_embedding(updated_metadata, cchunk.text, component_type="code")
                    else:
                        sm_code = cchunk.text
                    code_small_chunks.append(Document(page_content = sm_code, metadata=updated_metadata))
                else:
                    messages = [
                        {"role": "system", "content": "Bạn là chuyên gia ngôn ngữ có khả năng tổng hợp thông tin từ đoạn code để tạo context truy vấn cho ứng dụng RAG."},
                        {"role": "user", "content": f"Hãy viết một đoạn mô tả đoạn code sau {cchunk.text} đảm bảo các yếu tố sau:\n 1) Câu trả lời bằng tiếng Việt.\n 2) Nội dung cô động, đảm bảo trích xuất được các từ khóa chính (keyword) phục vụ cho việc tìm kiếm truy vấn (query search).\n 3) Câu trả lời chỉ có đoạn mô tả bảng bằng text không có thêm phần giải thích, hoặc ký hiệu icon lạ,.."},
                    ]
                    sm_code = self.chat_with_llm(messages)

                    # updated_metadata["type"] = "summary"
                    # updated_metadata["code"] = cchunk.text
                    updated_metadata.update({"full_context": cchunk.text})
                    if self.append_code_header:
                        sm_code = self.prepare_for_embedding(updated_metadata, sm_code, component_type="code")
                    else:
                        sm_code = sm_code
                    code_small_chunks.append(Document(page_content = sm_code, metadata=updated_metadata))
        return code_small_chunks

    def table_chunking(self, content, metadata):
        #chunk table
        table_small_chunks = []
        for table in content:
            table_content = table.content
            start_index=table.start_index
            end_index=table.end_index

            #whether split table or not
            if self.split_table:
                table_content: list = self.chunker_table(table_content)
                for tb in table_content:
                    updated_metadata = copy.deepcopy(metadata)
                    updated_metadata.update({"type": "table", "full_context":tb.text})
                    if self.use_summarize_table:
                        #get context of table
                        context_tabel = tb.text
                        #get summarize for table
                        messages = [
                            {"role": "system", "content": "Bạn là chuyên gia ngôn ngữ có khả năng tổng hợp thông tin từ bảng biểu để tạo context truy vấn cho ứng dụng RAG."},
                            {"role": "user", "content": f"Hãy viết một đoạn mô tả bảng sau {context_tabel} đảm bảo các yếu tố sau:\n 1) Câu trả lời bằng tiếng Việt.\n 2) Nội dung cô đọng, ngắn gọn, đảm bảo trích xuất được các từ khóa chính (keyword) phục vụ cho việc tìm kiếm truy vấn (query search).\n 3) Câu trả lời chỉ có đoạn mô tả bảng bằng text không có thêm phần giải thích, hoặc ký hiệu icon lạ,.."},
                        ]
                        sm_table = self.chat_with_llm(messages)
                    else:
                        sm_table = tb.text
                    
                    if self.append_table_header:
                        sm_table = self.prepare_for_embedding(updated_metadata, sm_table, component_type="table")
                    table_small_chunks.append(Document(page_content = sm_table, metadata=updated_metadata))
            else:
                updated_metadata = copy.deepcopy(metadata)
                updated_metadata.update({"type": "table", "full_context":table_content})
                if self.use_summarize_table:
                    #get context of table
                    context_tabel = table_content
                    #get summarize for table
                    messages = [
                        {"role": "system", "content": "Bạn là chuyên gia ngôn ngữ có khả năng tổng hợp thông tin từ bảng biểu để tạo context truy vấn cho ứng dụng RAG."},
                        {"role": "user", "content": f"Hãy viết một đoạn ngắn mô tả bảng sau {context_tabel} đảm bảo các yếu tố sau:\n 1) Câu trả lời bằng tiếng Việt.\n 2) Nội dung cô đọng, ngắn gọn, đảm bảo trích xuất được các từ khóa chính (keyword) phục vụ cho việc tìm kiếm truy vấn (query search).\n 3) Câu trả lời chỉ có đoạn mô tả bảng bằng text không có thêm phần giải thích, hoặc ký hiệu icon lạ,.."},
                    ]
                    sm_table = self.chat_with_llm(messages)
                else:
                    sm_table = table_content
                
                if self.append_table_header:
                    sm_table = self.prepare_for_embedding(updated_metadata, sm_table, component_type="table")

                table_small_chunks.append(Document(page_content = sm_table, metadata=updated_metadata))
        return table_small_chunks

    def prepare_for_embedding(self, metadata, raw_text, component_type="text"):
       
        section_path = [metadata.get(f"Header {i}") for i in range(1, 5)]
        section_path = [s for s in section_path if s]
        section = " > ".join(section_path)

        formatted_text = (
            f"Section: {section}\n"
            f"Type: {component_type}\n"
            f"Content:\n{raw_text}"
        )
        return formatted_text

    def build_and_persist_vectorstore(
        self,
        chunks: Optional[List[Document]]=None,
        path_csv = None,
        metadatas: Optional[List[Dict]] = None,
        model_name: str = EMBED_MODEL,
        persist_dir: str = PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ):
        """
        - chunks: list of Document
        - metadatas: optional list of dicts, same length as chunks
        """
        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError("metadatas must be None or have same length as chunks")
        
        # from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
        from sentence_transformers import SentenceTransformer
        import ast

        # hf_emb = GeminiEmbeddings(GEMINI_API_KEY, model=model_name)
        hf_emb = HuggingFaceEmbeddings(model_name=model_name)

        if path_csv is not None:
            chunks = []
            df = pd.read_csv(path_csv)
            for i in range(len(df)):
                row = df.iloc[i]
                # print(type(row["metadata"]))
                chunks.append(Document(page_content=row["page_content"], metadata=ast.literal_eval(row["metadata"])))

        import shutil
        chroma_path = os.path.join(persist_dir, collection_name)
        if os.path.exists(chroma_path):
            print(f"Delete ChromaDB: {chroma_path}")
            shutil.rmtree(chroma_path)

        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=hf_emb,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )

        vectordb.persist()
        return vectordb

    def semantic_search(self, vectordb: Chroma, query: str, k: int = 5):
        results = vectordb.similarity_search_with_score(query, k=k)
        return results


if __name__ == "__main__":
    processor = DocumentProcesser(path = "/data/AIRACE/arxiv_rag/output/paper/paper.md",\
                                   detect_language=True, split_table=False, \
                                    chunk_size=512, table_size=2048, use_summarize_code=True,\
                                     use_summarize_table=True, chat_with_llm=chat_with_llm)
   
    chunk = processor.process()
    data_frame = {"page_content":[], "metadata":[]}
    for i, c in enumerate(chunk):
        # print(f"================== chunk: {i} --> chunk size: {len(c.page_content)} ==================")
        # print(c)
        data_frame["page_content"].append(c.page_content)
        data_frame["metadata"].append(c.metadata)

    df = pd.DataFrame(data_frame)
    df.to_csv("/data/AIRACE/arxiv_rag/output/save_chunk.csv")



    # print("\n\n --------------------------- \n\n")
    vectordb = processor.build_and_persist_vectorstore(path_csv="/data/AIRACE/arxiv_rag/output/save_chunk.csv")
    q = "what is the mAP of Yolov8 model?"
    hits = processor.semantic_search(vectordb, q, k=20)
    for i, doc in enumerate(hits):
        print(f"================ chunk {i} -> score: {doc[1]} ==================")
        print(i+1, doc[0].metadata, doc[0].page_content)
        



        