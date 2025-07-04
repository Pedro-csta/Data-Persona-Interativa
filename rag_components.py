# rag_components.py
import pandas as pd
from langchain.docstore.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        if "text" not in df.columns or "product" not in df.columns:
            raise ValueError("O CSV deve conter as colunas 'text' e 'product'.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Arquivo não encontrado em: {filepath}. Certifique-se que ele existe.")

def create_rag_chain(dataframe, product_name, persona_name, api_key):
    product_df = dataframe[dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty:
        return None
    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.3)
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
    ---
    Sua Resposta (como {persona_name}):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return rag_chain

def generate_suggested_questions(rag_chain, persona_name):
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
