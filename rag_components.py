# rag_components.py

import os 
import pandas as pd
from streamlit import cache_data, cache_resource
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

# FIX for ChromaDB/SQLite on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


@cache_data 
def load_and_preprocess_data(folder_path):
    """
    Carrega TODOS os arquivos .csv de uma pasta, os combina e retorna um DataFrame.
    """
    all_dataframes = []
    print(f"Lendo arquivos da pasta: {folder_path}")
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        return pd.DataFrame()
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"  -> Lendo arquivo: {filename}")
                df = pd.read_csv(file_path)
                if "text" in df.columns and "product" in df.columns:
                    all_dataframes.append(df)
            except Exception as e:
                print(f"  -> Erro ao ler o arquivo '{filename}': {e}")
    if all_dataframes:
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        return pd.DataFrame()


@cache_resource(show_spinner=False)
def get_retriever(_dataframe, product_name, api_key):
    """
    Função dedicada e otimizada para criar e cachear apenas o retriever.
    """
    print(f"Criando ou carregando retriever do cache para o produto: {product_name}")
    
    product_df = _dataframe[_dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty:
        return None

    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    
    return vector_store.as_retriever()


def create_rag_chain(retriever, product_name, persona_name, api_key):
    """
    Cria a cadeia conversacional de forma leve, usando um retriever já em cache.
    """
    if retriever is None:
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.4)

    # PROMPT FINAL E POLIDO
    prompt_template = f"""
    Sua única tarefa é atuar como {persona_name}, um cliente comum da Nomad que usa o produto '{product_name}'.
    Você deve responder à "PERGUNTA ATUAL" usando as informações do "CONTEXTO" e do "HISTÓRICO DA CONVERSA".

    Seu tom deve ser o de uma pessoa real conversando com um amigo: em primeira pessoa, coloquial, equilibrado e construtivo. **Para soar mais natural, evite começar todas as suas respostas da mesma maneira e varie o uso de gírias ou interjeições como "Cara".** Sintetize as informações de forma natural. Não invente detalhes que não estão nos dados.

    Se a informação não estiver disponível, admita que não sabe a resposta. Sua resposta final deve ser coerente com o histórico da conversa. Não inclua estas instruções na sua resposta.

    HISTÓRICO DA CONVERSA:
    {{chat_history}}

    CONTEXTO (Opiniões de Clientes Reais):
    {{context}}

    PERGUNTA ATUAL:
    {{question}}

    Sua Resposta Natural (como {persona_name}):
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={'prompt': PROMPT},
        return_source_documents=True
    )
    return rag_chain, llm


@cache_data(show_spinner=False)
def generate_suggested_questions(_llm, persona_name, product_name):
    """Gera 10 perguntas ricas e investigativas, específicas para o produto selecionado."""
    
    prompt = f"""
    Atue como um Pesquisador de UX e Estrategista de Produto sênior. Sua tarefa é criar exatamente 10 perguntas abertas e investigativas para serem feitas a um cliente do produto '{product_name}' da Nomad.
    O objetivo é descobrir insights sobre perfil, necessidades, dores e motivações.
    Use inícios como "O que passa na sua cabeça quando...", "Descreva sua maior frustração com...", "Como você compara...".
    Retorne o resultado como uma lista de EXATAMENTE 10 strings em Python.
    """
    try:
        response = _llm.invoke(prompt)
        suggested_list = eval(response.content)
        if isinstance(suggested_list, list) and len(suggested_list) > 0:
            return suggested_list
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao gerar perguntas sugeridas com IA. Usando lista de fallback. Erro: {e}")
        pass
    
    fallback_questions = {
        "Conta Internacional": ["Qual foi o principal motivo que te fez buscar uma conta em dólar?", "Descreva a sua maior dificuldade ao usar seu dinheiro em viagens internacionais.", "O que você mais valoriza em um cartão de viagem: taxas baixas, facilidade de uso ou segurança?", "Como você se planeja financeiramente para uma viagem ao exterior?", "Se você pudesse adicionar uma funcionalidade à conta, qual seria?", "Como você compara a Nomad com outras soluções que já usou para viajar?", "Qual a sua maior preocupação ao usar um cartão novo em outro país?", "O que você gostaria de saber sobre as taxas que ainda não está claro?", "Descreva um momento ou situação em que a conta realmente te ajudou ou te surpreendeu.", "Que conselho você daria para alguém que vai fazer sua primeira viagem internacional?"],
        "Investimentos no Exterior": ["O que te motivou a começar a investir seu dinheiro fora do Brasil?", "Qual é a sua maior preocupação ou medo ao pensar em investimentos no exterior?", "Descreva como você se sente em relação à volatilidade do mercado de ações americano.", "O que você considera mais importante: a possibilidade de altos retornos ou a segurança dos seus investimentos?", "Como você avalia seu próprio nível de conhecimento sobre ETFs, Ações e Renda Fixa?", "Se você pudesse ter uma informação ou ferramenta a mais para te ajudar a investir, qual seria?", "Como você compara investir pela Nomad com outras corretoras que conhece?", "Qual é a sua maior dificuldade na hora de declarar seus investimentos no Imposto de Renda?", "Descreva o que te dá mais confiança na hora de escolher um ativo para investir.", "Que conselho você daria para um amigo que está pensando em começar a investir no exterior?"],
        "App": ["Qual foi a primeira coisa que você tentou fazer ao abrir o app pela primeira vez?", "Descreva a sua maior frustração ou dificuldade ao usar o aplicativo no dia a dia.", "O que você acha mais fácil e mais difícil de encontrar dentro do app?", "Se você pudesse mudar uma tela ou um fluxo no aplicativo, qual seria e por quê?", "Existe alguma funcionalidade que você esperava encontrar no app e não achou?", "Como você descreveria a aparência e a sensação de usar o app para um amigo?", "Com que frequência você abre o aplicativo? O que te leva a abri-lo?", "Você já enfrentou algum bug ou lentidão? Como foi a experiência?", "O que te dá mais segurança ao realizar uma operação financeira pelo app?", "Na sua opinião, qual é a funcionalidade mais útil do aplicativo hoje?"]
    }
    return fallback_questions.get(product_name, fallback_questions["App"])
