from typing import TypedDict, Optional, List
from tools.classify_scope import Scope, classify_scope
from tools.search_company_documents import ChunkResultado, create_search_company_documents
from tools.sintetizar_resposta import sintetizar_resposta
from tools.check_grounding import check_grounding
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    """
    O estado funciona como uma memória compartilhada.
  	Cada nó pode ler informações existentes e adicionar ou atualizar novos campos.
    """
    query: str
    scope: Optional[Scope]           # resultado do classify_scope
    chunks: List[ChunkResultado]     # resultado do retrieval
    resposta: Optional[str]
    fundamentada: Optional[bool]     # resultado do check_grounding

def build_graph(llm, vector_store) -> CompiledStateGraph:
    search_company_documents = create_search_company_documents(vector_store)

    def node_classify_scope(state: AgentState) -> dict:
        scope = classify_scope(state["query"], llm)
        return {"scope": scope}

    def node_search(state: AgentState) -> dict:
        scope = state["scope"]
        categorias = scope.categories if scope.confidence == "alta" else None
        chunks = search_company_documents.invoke({
            "query": state["query"],
            "categorias": categorias,
        })
        print(f"DEBUG chunks encontrados: {len(chunks)}")
        for c in chunks:
            print(f"  score={c.score:.3f} doc={c.documento}")
        return {"chunks": chunks}

    def node_synthesize(state: AgentState) -> dict:
        resposta = sintetizar_resposta(state["query"], state["chunks"], llm)
        return {"resposta": resposta}

    def node_check_grounding(state: AgentState) -> dict:
        resultado = check_grounding(state["resposta"], state["chunks"], llm)
        print(f"DEBUG fundamentada={resultado.fundamentada} justificativa={resultado.justificativa}")
        return {"fundamentada": resultado.fundamentada}

    def node_refuse_out_of_scope(state: AgentState) -> dict:
        return {"resposta": "Isso está fora do que consigo ajudar por aqui..."}

    def node_refuse_no_grounding(state: AgentState) -> dict:
        return {"resposta": "Não tenho essa informação na minha base de conhecimento..."}

    # Montar grafo
    workflow = StateGraph(AgentState)

    workflow.add_node('classify_scope', node_classify_scope)
    workflow.add_node('search_company_documents', node_search)
    workflow.add_node("synthesize", node_synthesize)
    workflow.add_node("check_grounding", node_check_grounding)
    workflow.add_node("refuse_out_of_scope", node_refuse_out_of_scope)
    workflow.add_node("refuse_no_grounding", node_refuse_no_grounding)

    workflow.add_edge(START, "classify_scope")

    def route_scope(state: AgentState) -> str:
        return "search" if state["scope"].in_scope else "out_of_scope"
    
    workflow.add_conditional_edges(
        "classify_scope",
        route_scope,
        {"search": "search_company_documents", "out_of_scope": "refuse_out_of_scope"}
    )

    def route_chunks(state: AgentState) -> str:
        return "synthesize" if state["chunks"] else "no_grounding"

    workflow.add_conditional_edges(
        "search_company_documents",
        route_chunks,
        {"synthesize": "synthesize", "no_grounding": "refuse_no_grounding"}
    )

    workflow.add_edge("synthesize", "check_grounding")

    def route_grounding(state: AgentState) -> str:
        return "ok" if state["fundamentada"] else "no_grounding"

    workflow.add_conditional_edges(
        "check_grounding",
        route_grounding,
        {"ok": END, "no_grounding": "refuse_no_grounding"},
    )
    workflow.add_edge("refuse_out_of_scope", END)
    workflow.add_edge("refuse_no_grounding", END)

    graph = workflow.compile()

    return graph