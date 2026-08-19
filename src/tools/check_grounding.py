from typing import List
from tools.search_company_documents import ChunkResultado
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

class GroundingCheck(BaseModel):
    fundamentada: bool = Field(
        description="True somente se TODAS as afirmações da resposta estão "
        "apoiadas no contexto. False se qualquer parte não tiver base explícita."
    )
    trechos_nao_apoiados: List[str] = Field(
        default_factory=list,
        description="Trechos da resposta que não encontram respaldo direto no contexto."
    )
    justificativa: str = Field(description="Breve explicação do veredito.")

prompt = ChatPromptTemplate.from_template("""
Você verifica se uma resposta usa exclusivamente informações presentes no contexto,
sem adicionar, inferir além do escrito ou combinar dados de forma não explícita.

Resposta a verificar: 
{resposta}

Contexto disponível:
{contexto}

Avalie cada afirmação da resposta individualmente contra o contexto.
""")

def check_grounding(answer: str, chunks: List[ChunkResultado], llm) -> GroundingCheck:
    if not chunks:
        return GroundingCheck(
            fundamentada=False,
            trechos_nao_apoiados=[answer],
            justificativa="Nenhum contexto disponível para a verificação."
        )
    
    contexto = "\n\n".join(chunk.conteudo for chunk in chunks)
    structured_llm = llm.with_structured_output(GroundingCheck)
    cadeia = prompt | structured_llm
    resposta = cadeia.invoke({"resposta": answer, "contexto": contexto})
    return resposta