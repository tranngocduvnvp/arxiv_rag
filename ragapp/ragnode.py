from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
import os
from retrival_app import fuse_retrival_rerank, vectorstore, bm25_retriever
from langchain.schema import Document
from langgraph.graph import END, StateGraph

# define llm
llm = ChatOpenAI(
    api_key="AIzaSyCwDT5DA",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.0-flash"
)
web_search_tool = TavilySearchResults()

# define llm_chain
preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise."""
prompt_llm_chain = ChatPromptTemplate.from_messages(
    [
        ("system", preamble),
        ("human", "Question: {question}\nAnswer: ")
    ]
)
llm_chain = prompt_llm_chain | llm | StrOutputParser()


# define rag_chain
RAG_PREAMBLE =  """
You are an AI assistant that must answer user questions strictly based on the provided context. 
Follow these rules:

1. Use only the information contained in the supplied context to answer.
2. If the context does not contain the information needed to answer the question, respond exactly with: "I don't know."
3. Do not fabricate facts, infer missing details, or use outside knowledge.
4. Keep answers concise, accurate, and directly related to the question.
5. If the user asks multiple questions, answer only those that can be supported by the context; otherwise reply "I don't know."

Context:
{{context}}

User question:
{{question}}
"""

rag_prompt = PromptTemplate(
    template=RAG_PREAMBLE,
    input_variables=["context", "question"]
)
rag_chain = (
    rag_prompt
    | llm
    | StrOutputParser()
)

# define tool
class Generation(BaseModel):
    '''
    Taọ ra câu trả lời từ các tri thức có sẵn của bạn
    '''
    query: str = Field(description="Câu truy vấn của người dùng")

class Retrieval(BaseModel):
    '''
    Truy truy vấn từ trong database để lấy các thông tin của tài liệu, từ đó cung cấp context cho LLM
    '''
    query: str = Field(description="Câu truy vấn dùng để seach trong vectorstore")

# define output schema
class GradeRespone(BaseModel):
    '''
    Điểm số Binary dùng để đánh giá câu trả lời và truy vấn có liên quan với nhau hay không
    '''
    binary_score: str = Field(description="Respone are relevant to the question, 'yes' or 'no'")

# define websearch
class WebSearch(BaseModel):
    '''
    Seach web để tìm nội dung của câu trả lời nếu trong câu hỏi người dùng có nói đến việc search tài liệu trên web
    '''
    query: str = Field(description="Câu truy vấn để tìm thông tin trên web")

#================== binding tool router =========================:
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Bạn là một chuyên gia router có chức năng điều hướng câu hỏi. Nếu câu hỏi mang tính chất chung thì điều hướng đến tool Generation, Nếu câu hỏi liên quan đến kỹ thuật AI hoặc paper thì điều hướng đến tool RetrivalARXIV"),
        ("Human", "{question}")
    ]
)
llm_router = llm.bind_tools(tools=[WebSearch, Retrieval])
question_router = router_prompt | llm_router

#=================== binding output structure ====================:
llm_grade_document_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Bạn là một chuyên gia đánh giá độ liên của tài câu trả lời được cung cấp với câu hỏi của người dùng.\n"
        "Nếu câu hỏi liên quan về mặt ngữ nghĩa với câu trả lời được cung cấp, đánh giá là có liên quan. Trong trường hợp khác đánh giá là không liên quan.\n"
        "Cho điểm số ở dạng Binary là 'yes' hoặc 'no'"),
        ("human", "Respone: \n\n {document} \n\n User question: {question}")
    ]
)
llm_grade_document_query = llm.with_structured_output(GradeRespone)
grade_answer_query_chain = llm_grade_document_prompt | llm_grade_document_query


#==================== Make stage for graph ========================:
class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]
    tool: str


#======================= Define node ==============================
def llm_generation(state):
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"generation": generation}


def rag_retrival(state):
    question = state["question"]
    document_retrievaled = fuse_retrival_rerank(question, None, None,\
                                                 vectorstore, bm25_retriever, 5, 0.99, is_ranking=False)
    
    for i, chunk in enumerate(document_retrievaled):
      append_context =  chunk["metadata"].get("full_context","")
      context_chunks +=f'[chunk {i+1}]\n{chunk["document"]}\n{append_context}'
      context_chunks +="\n\n -------------\n\n"
      if i+1 == 5:
          break 
    document_context = Document(page_content=context_chunks)
    return {"documents": document_context, "tool": "rag_retrival"}
    

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "tool": "web_search"}



def rag_generation(state):
    question = state["question"]
    context_chunks = state["documents"]
    prompt_RAG = rag_chain.invoke({"context": context_chunks, "question":question})
    return {"generation": prompt_RAG, "question": question}


#=========================== define edge ==============================

def route_question(state):
    question = state["question"]
    source = llm_router.invoke({"question": question})

    # Fallback to LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_generation"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source"
    
    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "WebSearch":
        # todo: The question doesn't need to tool call
        return "web_search"
    elif datasource == "Retrieval":
        # to do: The question need to call call
        return "rag_retrival"
    
def grade_answer_query(state):
    question = state["question"]
    answer = state["generation"]
    grade = grade_answer_query_chain.invoke({"document": answer, "question": question})
    if grade == "yes":
        #to do
        return "useful"
    else:
        #to do check tool use
        if state["tool"] == "rag_retrival":  
          return "web_search"
        elif state["tool"] == "web_search":
          return "rag_retrival"    
    


# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node(llm_generation)
workflow.add_node(web_search)
workflow.add_node(rag_retrival)
workflow.add_node(rag_generation)

# Define graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "llm_generation": llm_generation,
        "web_search": web_search,
        "rag_retrival": rag_retrival
    }
)

workflow.add_edge("web_search", "rag_generation")
workflow.add_edge("rag_retrival", "rag_generation")

workflow.add_conditional_edges(
    "rag_generation",
    grade_answer_query,
    {
        "useful": END,
        "web_search": "web_search",
        "rag_retrival": "rag_retrival"
    }
)
workflow.add_edge("llm_generation", END)
app = workflow.compile()