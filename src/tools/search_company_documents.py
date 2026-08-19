from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, List

class ChunkResultado(BaseModel):
    conteudo: str
    documento: str
    numero_secao: str
    titulo_secao: str
    score: float

def create_search_company_documents(vector_store, score_threshold: float = 0.5):
    def _buscar(query: str, categorias: Optional[List[str]] = None, k: int = 5) -> List[ChunkResultado]:
        filtro = {"categoria": {"$in": categorias}} if categorias else None
        print(f"DEBUG filtro aplicado: {filtro}")
        resultados = vector_store.similarity_search_with_relevance_scores(
            query, k=k, filter=filtro
        )

        return [
            ChunkResultado(
                conteudo=doc.page_content,
                documento=doc.metadata.get("documento", "desconhecido"),
                numero_secao=doc.metadata.get("numero_secao", ""),
                titulo_secao=doc.metadata.get("titulo_secao", ""),
                score=score,
            )
            for doc, score in resultados
            if score >= score_threshold
        ]

    @tool
    def search_company_documents(query: str, categorias: Optional[List[str]] = None) -> List[ChunkResultado]:
        """
        Busca trechos relevantes nos documentos internos da BimBam Buy
        (envio, garantia, pagamento, afiliados, reembolsos).

        Use 'categorias' com os valores retornados por classify_scope quando a
        confiança da classificação for alta, para restringir a busca ao(s)
        documento(s) certo(s). Deixe None (ou passe múltiplas categorias) quando
        a pergunta cruzar temas ou a confiança for baixa.

        Retorna uma lista vazia se nada acima do limiar de relevância for
        encontrado — isso deve ser tratado como 'sem base documental para
        responder', não como sinal para o LLM improvisar.
        """
        return _buscar(query, categorias=categorias)

    return search_company_documents