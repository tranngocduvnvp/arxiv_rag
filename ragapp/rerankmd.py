# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs, max_length, prefix_tokens, suffix_tokens, device, tokenizer):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(device)
    return inputs

@torch.no_grad()
def compute_logits(model, inputs, token_true_id, token_false_id, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores




def ranking(query, documents, model, tokenizer):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
    device = model.device
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)    
    task = 'Given a web search query, retrieve relevant passages that answer the query'

    queries = [query]*len(documents)

    pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

    # Tokenize the input texts
    inputs = process_inputs(pairs, max_length, prefix_tokens, suffix_tokens, device, tokenizer)
    scores = compute_logits(model, inputs, token_true_id, token_false_id)

    return inputs, scores


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()

    documents = [
        "Việt Nam là một quốc gia đông nam á có 34 tỉnh thành",
        "Việt Nam có Quang Hải là một cầu thủ đá bóng giỏi",
    ]

    inputs, scores = ranking("số tỉnh thành của việt nam là bao nhiêu", documents, model, tokenizer)
    print(scores)