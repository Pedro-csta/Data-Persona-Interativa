# rag_components.py

import os
import pandas as pd
from typing import TypedDict, List
from streamlit import cache_data, cache_resource

from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Importações principais do LangGraph
from langgraph.graph import StateGraph, END

# FIX for ChromaDB/SQLite on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# --- Definição do Estado do Grafo ---
# O estado é o objeto que flui entre os agentes, carregando todas as informações.
class AgentState(TypedDict):
    question: str
    chat_history: list
    product_name: str
    persona_name: str
    documents: List[Document]
    final_answer: str

# --- Funções e Classes para os Nós do Grafo ---

# Define a estrutura de saída do nosso primeiro agente
class DecomposedQuery(BaseModel):
    search_queries: List[str] = Field(description="Uma lista de 2 a 3 strings de busca otimizadas e específicas para encontrar a melhor informação na base de conhecimento.")

@cache_data
def load_and_preprocess_data(folder_path):
    all_dataframes = []
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError: return pd.DataFrame()

    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if "text" in df.columns and "product" in df.columns:
                    if filename == 'info_oficial.csv':
                        df['text'] = '[FONTE OFICIAL]: ' + df['text'].astype(str)
                    else:
                        df['text'] = '[OPINIÃO DE USUÁRIO]: ' + df['text'].astype(str)
                    all_dataframes.append(df)
            except Exception as e: print(f"Erro ao ler '{filename}': {e}")
    if all_dataframes: return pd.concat(all_dataframes, ignore_index=True)
    return pd.DataFrame()

@cache_resource(show_spinner=False)
def get_retriever(_dataframe, product_name, api_key):
    print(f"Carregando retriever para: {product_name}")
    product_df = _dataframe[_dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty: return None
    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    return vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 10, 'fetch_k': 30})

# --- NÓ 1: O AGENTE ANALISTA DE CONSULTA ---
def query_analyzer_node(state: AgentState, llm_query_analyzer):
    print("--- Agente: Analista de Consulta ---")
    prompt = f"""
    Sua tarefa é atuar como um especialista em buscas. Analise a pergunta do usuário e o histórico da conversa para gerar de 2 a 3 variações de busca que sejam otimizadas para encontrar as informações mais relevantes em uma base de conhecimento.

    Exemplo:
    Pergunta: "Como as taxas da Nomad se comparam com as da Wise e qual a opinião dos usuários sobre isso?"
    Saída: ["taxas de serviço Nomad vs Wise", "opinião dos usuários sobre taxas e custos da Nomad", "vantagens e desvantagens das taxas da Nomad"]

    Histórico: {state['chat_history']}
    Pergunta do Usuário: {state['question']}
    """
    structured_llm = llm_query_analyzer.with_structured_output(DecomposedQuery)
    response = structured_llm.invoke(prompt)
    print(f"-> Buscas geradas: {response.search_queries}")
    return {"documents": [], "search_queries": response.search_queries}

# --- NÓ 2: O AGENTE PESQUISADOR ---
def retrieval_node(state: AgentState, retriever):
    print("--- Agente: Pesquisador ---")
    all_docs = []
    # Usa as buscas geradas pelo agente anterior para pesquisar na base
    for query in state["search_queries"]:
        docs = retriever.invoke(query)
        all_docs.extend(docs)
    
    # Remove duplicatas
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    print(f"-> Documentos encontrados: {len(unique_docs)}")
    return {"documents": list(unique_docs)}

# --- NÓ 3: O AGENTE SINTETIZADOR (A PERSONA) ---
def synthesis_node(state: AgentState, llm_synthesis):
    print("--- Agente: Sintetizador (Persona) ---")
    prompt_template = f"""
    Sua única tarefa é atuar como {state['persona_name']}, um cliente comum da Nomad que usa o produto '{state['product_name']}'.
    Você deve responder à "PERGUNTA ATUAL" usando as informações do "CONTEXTO" e do "HISTÓRICO DA CONVERSA".

    Seu tom deve ser o de uma pessoa real conversando com um amigo: em primeira pessoa, coloquial, equilibrado e construtivo.
    Sintetize as informações de forma natural. Não invente detalhes.
    Se a informação não estiver disponível, admita que não sabe. Sua resposta final deve ser coerente com o histórico.
    
    REGRA CRÍTICA - HIERARQUIA: Para fatos sobre o produto, priorize o contexto `[FONTE OFICIAL]`. Para experiências, use `[OPINIÃO DE USUÁRIO]`. Se houver conflito, comente sobre isso.

    HISTÓRICO DA CONVERSA: {state['chat_history']}
    CONTEXTO (Informações coletadas para você): {state['documents']}
    PERGUNTA ATUAL: {state['question']}
    Sua Resposta Natural e Coerente (como {state['persona_name']}):
    """
    response = llm_synthesis.invoke(prompt_template)
    print("-> Resposta final gerada.")
    return {"final_answer": response.content}


# --- FUNÇÃO PRINCIPAL QUE MONTA E RETORNA O GRAFO ---
def create_agentic_rag_app(retriever, api_key):
    if retriever is None: return None
    
    # Define os LLMs para cada agente
    llm_query_analyzer = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    llm_synthesis = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.4)
    
    # Montagem do Grafo
    workflow = StateGraph(AgentState)
    workflow.add_node("query_analyzer", lambda state: query_analyzer_node(state, llm_query_analyzer))
    workflow.add_node("retriever", lambda state: retrieval_node(state, retriever))
    workflow.add_node("synthesizer", lambda state: synthesis_node(state, llm_synthesis))

    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

# A função de gerar perguntas sugeridas permanece a mesma e não precisa do grafo
@cache_data(show_spinner=False)
def generate_suggested_questions(api_key, persona_name, product_name):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
    prompt = f"""
    Atue como um Pesquisador de UX. Crie 10 perguntas abertas para um cliente do produto '{product_name}' da Nomad para descobrir insights sobre perfil e dores. Retorne como uma lista Python.
    """
    try:
        response = llm.invoke(prompt)
        return eval(response.content)
    except:
        return ["Qual foi a primeira coisa que você tentou fazer no app?"]
