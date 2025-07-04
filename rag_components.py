# rag_components.py

import os 
import pandas as pd
from typing import TypedDict, List
from streamlit import cache_data, cache_resource

from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langgraph.graph import StateGraph, END

# FIX for ChromaDB/SQLite on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class AgentState(TypedDict):
    question: str
    chat_history: list
    product_name: str
    persona_name: str
    documents: List[Document]
    final_answer: str

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
                    if filename == 'info_oficial.csv': df['text'] = '[FONTE OFICIAL]: ' + df['text'].astype(str)
                    else: df['text'] = '[OPINIÃO DE USUÁRIO]: ' + df['text'].astype(str)
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
    return vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 8, 'fetch_k': 25})

def query_analyzer_node(state: AgentState, llm):
    print("--- Agente: Analista de Consulta ---")
    prompt = f"""Sua tarefa é atuar como um especialista em buscas. Analise a pergunta do usuário e o histórico da conversa para gerar de 2 a 3 variações de busca otimizadas.
    Histórico: {state['chat_history']}
    Pergunta do Usuário: {state['question']}"""
    class DecomposedQuery(BaseModel):
        search_queries: List[str] = Field(description="Uma lista de 2 a 3 strings de busca otimizadas.")
    structured_llm = llm.with_structured_output(DecomposedQuery)
    response = structured_llm.invoke(prompt)
    return {"documents": [], "search_queries": response.search_queries}

def retrieval_node(state: AgentState, retriever):
    print("--- Agente: Pesquisador ---")
    all_docs = []
    for query in state["search_queries"]:
        docs = retriever.invoke(query)
        all_docs.extend(docs)
    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
    return {"documents": list(unique_docs)}

def synthesis_node(state: AgentState, llm):
    print("--- Agente: Sintetizador (Persona) ---")
    prompt_template = f"""Sua única tarefa é atuar como {state['persona_name']}, um cliente comum da Nomad que usa o produto '{state['product_name']}'. Você deve responder à "PERGUNTA ATUAL" usando as informações do "CONTEXTO" e do "HISTÓRICO DA CONVERSA".
    Seu tom deve ser coloquial, equilibrado e construtivo. Varie o início das suas respostas. Sintetize as informações de forma natural. Se a informação não estiver disponível, admita que não sabe.
    REGRA CRÍTICA - HIERARQUIA: Para fatos sobre o produto, priorize o contexto `[FONTE OFICIAL]`. Para experiências, use `[OPINIÃO DE USUÁRIO]`. Se houver conflito, comente sobre isso.
    HISTÓRICO DA CONVERSA: {state['chat_history']}
    CONTEXTO: {state['documents']}
    PERGUNTA ATUAL: {state['question']}
    Sua Resposta Natural (como {state['persona_name']}):"""
    response = llm.invoke(prompt_template)
    return {"final_answer": response.content}

def create_agentic_rag_app(retriever, api_key):
    if retriever is None: return None
    llm_analyzer = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    llm_synthesizer = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.4)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("query_analyzer", lambda state: query_analyzer_node(state, llm_analyzer))
    workflow.add_node("retriever", lambda state: retrieval_node(state, retriever))
    workflow.add_node("synthesizer", lambda state: synthesis_node(state, llm_synthesizer))
    workflow.set_entry_point("query_analyzer")
    workflow.add_edge("query_analyzer", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()

# MUDANÇA: A função agora recebe a api_key e cria seu próprio LLM. Fica mais independente.
@cache_data(show_spinner=False)
def generate_suggested_questions(api_key, persona_name, product_name):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.5)
    prompt = f"""Atue como um Pesquisador de UX. Crie 10 perguntas abertas para um cliente do produto '{product_name}' da Nomad para descobrir insights sobre perfil e dores. Retorne como uma lista Python."""
    try:
        response = llm.invoke(prompt)
        return eval(response.content)
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao gerar perguntas sugeridas com IA. Usando fallback. Erro: {e}")
        pass
    fallback_questions = {
        "Conta Internacional": ["Qual foi o principal motivo que te fez buscar uma conta em dólar?", "Descreva a sua maior dificuldade ao usar seu dinheiro em viagens internacionais.", "O que você mais valoriza em um cartão de viagem: taxas baixas, facilidade de uso ou segurança?", "Como você se planeja financeiramente para uma viagem ao exterior?", "Se você pudesse adicionar uma funcionalidade à conta, qual seria?", "Como você compara a Nomad com outras soluções que já usou para viajar?", "Qual a sua maior preocupação ao usar um cartão novo em outro país?", "O que você gostaria de saber sobre as taxas que ainda não está claro?", "Descreva um momento ou situação em que a conta realmente te ajudou ou te surpreendeu.", "Que conselho você daria para alguém que vai fazer sua primeira viagem internacional?"],
        "Investimentos no Exterior": ["O que te motivou a começar a investir seu dinheiro fora do Brasil?", "Qual é a sua maior preocupação ou medo ao pensar em investimentos no exterior?", "Descreva como você se sente em relação à volatilidade do mercado de ações americano.", "O que você considera mais importante: a possibilidade de altos retornos ou a segurança dos seus investimentos?", "Como você avalia seu próprio nível de conhecimento sobre ETFs, Ações e Renda Fixa?", "Se você pudesse ter uma informação ou ferramenta a mais para te ajudar a investir, qual seria?", "Como você compara investir pela Nomad com outras corretoras que conhece?", "Qual é a sua maior dificuldade na hora de declarar seus investimentos no Imposto de Renda?", "Descreva o que te dá mais confiança na hora de escolher um ativo para investir.", "Que conselho você daria para um amigo que está pensando em começar a investir no exterior?"],
        "App": ["Qual foi a primeira coisa que você tentou fazer ao abrir o app pela primeira vez?", "Descreva a sua maior frustração ou dificuldade ao usar o aplicativo no dia a dia.", "O que você acha mais fácil e mais difícil de encontrar dentro do app?", "Se você pudesse mudar uma tela ou um fluxo no aplicativo, qual seria e por quê?", "Existe alguma funcionalidade que você esperava encontrar no app e não achou?", "Como você descreveria a aparência e a sensação de usar o app para um amigo?", "Com que frequência você abre o aplicativo? O que te leva a abri-lo?", "Você já enfrentou algum bug ou lentidão? Como foi a experiência?", "O que te dá mais segurança ao realizar uma operação financeira pelo app?", "Na sua opinião, qual é a funcionalidade mais útil do aplicativo hoje?"]
    }
    return fallback_questions.get(product_name, fallback_questions["App"])
