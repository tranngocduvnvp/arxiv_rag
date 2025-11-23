# Parse docutment
B1: parse_document/full_pipeline.py
B2: parse_document/main_merge_page.py
B3: parse_document/concat.py

# RAG
B1: Start server LLM
python model/serve_model.py

B2: Tiền xử lý bảng
python preprocess_table.py

B3: CHunking (sửa file đầu vào)
python utils/document_chunking.py

B4: retrival_ranking
python ragapp/retrival_app.py

B5: generation
python ragapp/generation.py