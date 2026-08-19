from components.vector_store import VectorStore
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.globals import set_debug, set_verbose
from workflow import build_graph
import gradio as gr
from functools import partial

import os

set_debug(True)
load_dotenv()

def run_graph(app, user_query: str) -> str:
    """
    Executes LangGraph with user query and return final answer
    """
    try:
        result = app.invoke({"query": user_query})
        return result["resposta"]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Erro: {e}"

def main():
    # 1. Initializes vector store
    vector_store = VectorStore()
    vector_store.create()

    print("Total de docs na collection:", len(vector_store.vectorstore.get()['documents']))
    if vector_store.count() == 0:
        vector_store.load()

    # 2. Call LLM
    llm = ChatGoogleGenerativeAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("GEMINI_MODEL")
    )

    # 3. Build tools
    graph = build_graph(llm, vector_store)

    # 4. Build Gradio
    iface = gr.Interface(
        fn=partial(run_graph, graph),
        inputs=gr.Textbox(label='Digite sua pergunta: '),
        outputs=gr.Markdown(label='Resposta final'),
        title='Agent BimBam Buy',
        description="Faça uma pergunta e obtenha uma resposta com fontes da empresa."
    )

    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()