import json
from openai import OpenAI
import pandas as pd
import re
import time
import uuid
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig, serve

file_save_retrieval = "/data/AIRACE/RAG/output/file_save_retrieval.json"
with open(file_save_retrieval, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Số lượng entry:", len(dataset))

def chat_with_llm(messages, base_url="http://0.0.0.0:23333/v1"):
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    completion = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=messages,
    )
    return completion.choices[0].message.content


def run_qwen3_4b_instruct_serve(
                        model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
                        ):
    
    backend_config = TurbomindEngineConfig(dtype="float16",tp=1)
    pipe = pipeline(model_name, backend_config=backend_config)
    return pipe

if __name__ == "__main__":

    pipe = run_qwen3_4b_instruct_serve()
    print("Load pipeline successfully!!!")
    number_context = 5
    df_ans = {"ques":[], "ans":[]}
    for nq, item in enumerate(dataset):
        query = item["query"]
        chunks = item["chunks"]
        answer = item["answer"]

        prompt = """
You are an assistant whose ONLY source of information is the text provided in the section "Các đoạn văn liên quan (đã xếp hạng)". 
You MUST follow these rules exactly.

RULES (must follow exactly):
1) Use **only** the provided chunks in "Các đoạn văn liên quan". Do not use any outside knowledge.
2) Always give step-by-step reasoning (minimum 2 numbered steps). When you use a chunk, state which one (e.g., "dựa vào chunk 1").
3) All arithmetic must be shown digit-by-digit if any; write out the calculation steps before any decimal result.
4) You MUST select exactly one of the four options A, B, C, or D provided in the "Đáp án" section. Do NOT invent options.
5) If none perfectly match, pick the one that is **most likely** based on the chunks and explain why.
6) You MUST NOT output any extra commentary, meta-text, or the phrase "Không tìm thấy đáp án".
7) OUTPUT FORMAT (absolute):
   - Line 1 must be: `Suy luận từng bước:`
   - Following lines: numbered steps (1., 2., ...).
   - Final line must contain **only** `[[Đáp án: <A|B|C|D>]]` and nothing else.
8) Do NOT repeat the question.

FEW-SHOT EXAMPLES (model should imitate these exactly):

Example correct output:
Suy luận từng bước:
1. Dựa vào chunk 1, ...
2. Dựa vào chunk 2, ...
3. Tổng hợp ..., suy ra đáp án là A.
[[Đáp án: A]]

Example INCORRECT outputs (you MUST NOT produce these):
- Any output that ends with "Không tìm thấy đáp án".
- Any output that prints extra lines after the final `[[Đáp án: ...]]`.
- Any output that prints anything other than the numbered steps + the final `[[Đáp án: ...]]`.

Now answer using only the supplied content.

### Câu hỏi:
{user_query}

### Các đoạn văn liên quan (đã xếp hạng):
{context}

### Đáp án (chỉ chọn 1 trong 4):
{answer}

**END INSTRUCTIONS.** Print the required output exactly as specified above and nothing else.
"""

        fix_prompt = regeneration_prompt = """
The previous response does not follow the required output format. 
Please **regenerate a completely new answer** that strictly follows the formatting rules below.

You MUST use the same information from the original question and related chunks — do NOT change the reasoning or content, only correct the format.

=========================
MANDATORY FORMAT RULES:
1. The first line MUST be exactly: `Suy luận từng bước:`
2. There must be at least two numbered reasoning steps: `1.`, `2.`, ...
3. The final line MUST be in this exact format: `[[Đáp án: <A|B|C|D>]]`
4. NOTHING should appear after that final line.
=========================

ORIGINAL QUESTION AND CONTEXT:
--------------------------------
### Question:
{user_query}

### Related passages (ranked):
{context}

### Possible answers (choose exactly one):
{answer}
--------------------------------

PREVIOUS INCORRECT RESPONSE:
--------------------------------
{previous_response}
--------------------------------

Now regenerate a NEW response that strictly follows the required format above.
"""

        
        context_chunks = ""
        for i, chunk in enumerate(chunks):
            # print("   Nội dung:", chunk["document"][:60], "...")
            # print("   Metadata:", chunk["metadata"])
            append_context =  chunk["metadata"].get("full_context","")
            context_chunks +=f'[chunk {i+1} - Độ liên quan dự tính: {chunk["rerank_score"]}]\n{chunk["document"]}\n{append_context}'
            context_chunks +="\n\n -------------\n\n"
            if i+1 == number_context:
                break 
        
        prompt = prompt.format(user_query=query, context=context_chunks, answer="\n".join(f"{key}: {value}" for key, value in answer.items()))
        
        print(f"\n\n\n============= Prompt {nq+1}===========>>>>\n\n")
        print(prompt)

        print(f"\n\n============= Respone {nq+1}===========>>>>\n\n")
        
        for nl in range(3):
            # reponse = chat_with_llm(prompt)
            reponse = pipe([prompt])[-1].text
            print(reponse)
            
            match = re.search(r"\[\[Đáp án:\s*([A-D])\]\]", reponse)
            if match:
                answer_choice = match.group(1)
                print(answer_choice)  # Output: B
                break
            else:
                answer_choice = "A"
                print("Không tìm thấy đáp án -> generate againt")
                prompt = fix_prompt.format(user_query=query, context=context_chunks, answer="\n".join(f"{key}: {value}" for key, value in answer.items()), previous_response=reponse)

        print(f"\n\n\n*************** Done Question {nq+1} *************\n\n\n")
        df_ans["ques"].append(nq+1)
        df_ans["ans"].append(answer_choice)

        # time.sleep(0.5)

        df_frame = pd.DataFrame(df_ans)
        df_frame.to_csv("/data/AIRACE/RAG/output/ans.csv", index=False)
        