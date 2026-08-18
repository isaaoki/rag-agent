from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal, List

CATEGORIAS = Literal["envio", "garantia", "pagamento", "afiliados", "reembolsos"]

class Scope(BaseModel):
    """
    Classifica se o prompt está dentro do domínio de suporte da BimBam Buy
    e, se estiver, quais documentos provavelmente contêm a resposta.
    """
    reason: str = Field(
        description="Breve justificativa para a classificação, "
        "incluindo por que esta ou não dentro do escopo."
    )

    in_scope: bool = Field(
        description="True se a pergunta pertence ao domínio de suporte da BimBam Buy "
        "(envio, garantia, pagamento, afiliados, reembolsos). False para qualquer "
        "pergunta fora desse domínio (conhecimento geral, outras empresas, etc)."
    )

    categories: List[CATEGORIAS] = Field(
        default_factory=list,
        description="Uma ou mais categorias relevantes para a pergunta, em ordem de "
        "relevância. Vazio se in_scope for False. Use mais de uma categoria quando a "
        "pergunta cruzar temas (ex: produto danificado no envio + garantia + reembolso)."
    )

    confidence: Literal["alta", "media", "baixa"] = Field(
        description="Confiança na classifcação. 'baixa' sinaliza ambiguidade que"
        "pode exigir busca ampla em vez de restrita por documento." 
    )

def classify_scope(query: str, llm):
    structured_llm = llm.with_structured_output(Scope)

    system_prompt = """Você classifica perguntas de clientes da BimBam Buy, uma
    empresa de e-commerce na América Latina.

    O domínio de suporte cobre exclusivamente:
    - envio: prazos, custos, rastreamento, incidências logísticas
    - garantia: falhas de produto, defeitos de fabricação, cobertura
    - pagamento: métodos de pagamento, cobranças, recusas, conciliação
    - afiliados: programa de afiliados, comissões, atribuição de vendas
    - reembolsos: devoluções, arrependimento, cancelamentos, reembolso de valores

    Se a pergunta não se relacionar claramente com nenhum desses temas — incluindo
    perguntas de conhecimento geral, sobre outras empresas, ou conversa casual —
    marque in_scope como False e categories como lista vazia. Não tente forçar um
    enquadramento."""

    return structured_llm.invoke([
        ("system", system_prompt),
        ("human", query),
    ])