from langchain.tools import tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
import os
from retrival_app import fuse_retrival_rerank, vectorstore, bm25_retriever
from langchain.schema import Document
import pprint
from langgraph.graph import END, StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


# os.environ["TAVILY_API_KEY"] = "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"
# define llm
llm = ChatOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    model="gemini-2.5-flash-lite"
)
web_search_tool = TavilySearchResults(max_results=2)

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
{context}

User question:
{question}
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
    """
    Generate an answer using the model's internal knowledge.
    """
    query: str = Field(description="The user's query to be answered using internal knowledge.")


class Retrieval(BaseModel):
    """
    Retrieve relevant information of paper from the database (vector store) to provide context for the LLM.
    """
    query: str = Field(description="The search query used to retrieve documents from the vector store.")


# define output schema
class GradeRespone(BaseModel):
    """
    A binary score used to evaluate whether the provided answer is relevant to the user question.
    """
    binary_score: str = Field(description="Indicates whether the response is relevant to the question: 'yes' or 'no'.")


# define websearch
class WebSearch(BaseModel):
    """
    Search the web to find information needed to answer the user question, especially if the question explicitly requests web-based information.
    """
    query: str = Field(description="The search query used to find information on the web.")


#================== binding tool router =========================:
router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert routing assistant responsible for deciding how a user question should be handled.\n"
            "- If the question is broad, general, or requires up-to-date external information, route it to the `WebSearch` tool.\n"
            "- If the question is technical, related to AI, research papers, or requires knowledge from Database, route it to the `Retrieval` tool.\n"
            "- For all other cases, answer the question directly using your internal knowledge without calling any tool."
        ),
        ("human", "{question}")
    ]
)

llm_router = llm.bind_tools(tools=[WebSearch, Retrieval])
question_router = router_prompt | llm_router

#=================== binding output structure ====================:
llm_grade_document_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator responsible for assessing whether the provided answer is semantically relevant to the user's question.\n"
            "If the answer is semantically related to the question, respond with a binary score of 'yes'.\n"
            "If the answer is not semantically related, respond with 'no'.\n"
            "Only return the binary score."
        ),
        (
            "human",
            "Response:\n\n{document}\n\nUser question:\n{question}"
        )
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
    print("--->>>> LLM GENERATION NODE ---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"generation": generation}


def rag_retrival(state):
    print("--->>>> RAG RETRIVAL NODE ---")
    question = state["question"]
    document_retrievaled = fuse_retrival_rerank(question, None, None,\
                                                 vectorstore, bm25_retriever, 5, 0.99, is_ranking=False)
    
    # print(question)
    # print(document_retrievaled)

    context_chunks = ""
    for i, chunk in enumerate(document_retrievaled):
      append_context =  chunk["metadata"].get("full_context","")
      context_chunks +=f'[chunk {i+1}]\n{chunk["document"]}\n{append_context}'
      context_chunks +="\n\n -------------\n\n"
      if i+1 == 5:
          break 
    
    print("context_chunks:\n", context_chunks)
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

    print("--->>>> WEB SEARCH NODE ---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    # print("web search result:\n",docs)
    web_results = "\n\n------------------------------\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    return {"documents": web_results, "tool": "web_search"}



def rag_generation(state):
    print("--->>>> RAG_GENERATION NODE ---")
    question = state["question"]
    context_chunks = state["documents"]
    result_RAG = rag_chain.invoke({"context": context_chunks, "question":question})
    # print("result_RAG:\n", result_RAG)
    return {"generation": result_RAG, "question": question}


#=========================== define edge ==============================

def route_question(state):
    question = state["question"]
    source = question_router.invoke({"question": question})

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
        print("---ROUTE WEB SEARCH---")
        return "web_search"
    elif datasource == "Retrieval":
        # to do: The question need to call call
        print("---ROUTE RETRIEVAL---")
        return "rag_retrival"
    

def grade_answer_query(state):
    question = state["question"]
    answer = state["generation"]
    grade = grade_answer_query_chain.invoke({"document": answer, "question": question})
    if grade.binary_score == "yes":
        #to do
        print("---USEFULL RESPONE---")
        return "useful"
    else:
        #to do check tool use
        if state["tool"] == "rag_retrival":
          print("---NOT USEFULL RESPONE USE WEB_SEARCH---")  
          return "web_search"
        elif state["tool"] == "web_search":
          print("---NOT USEFULL RESPONE USE RAG_RETRIEVAL---")  
          return "rag_retrival"    
    


# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("llm_generation",llm_generation)
workflow.add_node("web_search", web_search)
workflow.add_node("rag_retrival", rag_retrival)
workflow.add_node("rag_generation", rag_generation)

# Define graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "llm_generation": "llm_generation",
        "web_search": "web_search",
        "rag_retrival": "rag_retrival",
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


# Execute
inputs = {
    "question": "Let's search this database, let's show me what is key point of paper?"
}
for output in app.stream(inputs):
    for key, value in output.items():
        # Node
        pprint.pprint(f"Node '{key}':")
        # Optional: print full state at each node
    pprint.pprint("\n---\n")

# Final generation
pprint.pprint(value["generation"])