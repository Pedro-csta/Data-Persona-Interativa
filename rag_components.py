# rag_components.py

# Adicionamos a biblioteca 'os' para interagir com o sistema de arquivos
import os 
import pandas as pd
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# FIX for ChromaDB/SQLite on Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def load_and_preprocess_data(folder_path):
    """
    Carrega TODOS os arquivos .csv de uma pasta, os combina e retorna um DataFrame.
    """
    all_dataframes = []
    print(f"Lendo arquivos da pasta: {folder_path}")

    # Lista todos os arquivos no diretório fornecido
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"❌ Erro: A pasta '{folder_path}' não foi encontrada.")
        return pd.DataFrame() # Retorna um DataFrame vazio

    # Itera sobre cada arquivo encontrado
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"  -> Lendo arquivo: {filename}")
                df = pd.read_csv(file_path)
                # Verifica se as colunas necessárias existem
                if "text" in df.columns and "product" in df.columns:
                    all_dataframes.append(df)
                else:
                    print(f"  -> ⚠️ Aviso: O arquivo '{filename}' foi ignorado por não conter as colunas 'text' e 'product'.")
            except Exception as e:
                print(f"  -> ❌ Erro ao ler o arquivo '{filename}': {e}")
    
    # Combina todos os DataFrames lidos em um só
    if all_dataframes:
        print("✅ Arquivos combinados com sucesso!")
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        print("❌ Nenhum arquivo .csv válido foi encontrado ou lido.")
        return pd.DataFrame()


def create_rag_chain(dataframe, product_name, persona_name, api_key):
    """Cria e retorna uma cadeia RAG (RetrievalQA) configurada."""
    
    if dataframe.empty:
        print("❌ DataFrame vazio, não é possível criar a cadeia RAG.")
        return None

    # 1. Filtrar o DataFrame para o produto selecionado
    product_df = dataframe[dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty:
        return None 

    # 2. Converter as linhas do DataFrame para Documentos LangChain
    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]

    # 3. Inicializar os Embeddings do Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # 4. Criar o VectorStore (ChromaDB) em memória com os documentos
    vector_store = Chroma.from_documents(documents, embeddings)

    # 5. Inicializar o modelo LLM do Gemini
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.3)

    # 6. Criar o Template do Prompt
    prompt_template = f"""
    Você é {persona_name}, uma persona sintética que representa um cliente da marca Nomad para o produto '{product_name}'.
    Sua personalidade, opiniões e conhecimento são formados EXCLUSIVAMENTE pelo contexto de comentários e informações reais fornecidos abaixo.

    REGRAS IMPORTANTES:
    1. Responda sempre em primeira pessoa de forma amigável e coloquial, como um usuário real faria.
    2. NUNCA invente informações. Se o contexto não tiver a resposta, diga algo como "Poxa, sobre isso eu não tenho uma opinião formada ainda" ou "Não encontrei informações sobre esse ponto específico".
    3. Baseie-se SOMENTE no CONTEXTO fornecido. Não use nenhum conhecimento externo.
    4. Responda de forma concisa e direta.

    ---
    CONTEXTO:
    {{context}}
    ---

    PERGUNTA DO USUÁRIO:
    {{question}}

    Sua Resposta (como {persona_name}):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # 7. Criar a cadeia RetrievalQA
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )

    return rag_chain

def generate_suggested_questions(rag_chain, persona_name):
    """Gera perguntas sugeridas com base no contexto da cadeia RAG."""
    prompt = f"""
    Com base no seu conhecimento como {persona_name}, crie 3 perguntas curtas e diretas que um usuário interessado faria.
    Seu objetivo é iniciar uma conversa interessante.
    Retorne a resposta como uma lista de strings em Python. Exemplo: ['Pergunta 1?', 'Pergunta 2?', 'Pergunta 3?']
    """
    try:
        response = rag_chain.combine_documents_chain.llm_chain.llm.invoke(prompt)
        suggested_list = eval(response.content)
        if isinstance(suggested_list, list) and len(suggested_list) > 0:
            return suggested_list
    except:
        pass 
    
    return ["Quais são as principais tarifas?", "Como funciona na Europa?", "É seguro investir por aí?"]
