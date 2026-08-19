from tools.search_company_documents import ChunkResultado
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import List

prompt = ChatPromptTemplate.from_template("""
Sintetize uma resposta para a pergunta utilizando exclusivamente
as informações presentes no contexto.

Pergunta: {pergunta}

Contexto:
{contexto}
""")

def sintetizar_resposta(query: str, chunks: List[ChunkResultado], llm) -> str:
    contexto = "\n\n".join(chunk.conteudo for chunk in chunks)
    cadeia = prompt | llm | StrOutputParser()
    resposta = cadeia.invoke({"pergunta": query, "contexto": contexto})
    return resposta