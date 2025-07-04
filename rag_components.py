# rag_components.py

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
    all_dataframes = []
    print(f"Lendo arquivos da pasta: {folder_path}")
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"❌ Erro: A pasta '{folder_path}' não foi encontrada.")
        return pd.DataFrame()
    for filename in filenames:
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                print(f"  -> Lendo arquivo: {filename}")
                df = pd.read_csv(file_path)
                if "text" in df.columns and "product" in df.columns:
                    all_dataframes.append(df)
                else:
                    print(f"  -> ⚠️ Aviso: O arquivo '{filename}' foi ignorado por não conter as colunas 'text' e 'product'.")
            except Exception as e:
                print(f"  -> ❌ Erro ao ler o arquivo '{filename}': {e}")
    if all_dataframes:
        print("✅ Arquivos combinados com sucesso!")
        return pd.concat(all_dataframes, ignore_index=True)
    else:
        print("❌ Nenhum arquivo .csv válido foi encontrado ou lido.")
        return pd.DataFrame()


def create_rag_chain(dataframe, product_name, persona_name, api_key):
    if dataframe.empty:
        print("❌ DataFrame vazio, não é possível criar a cadeia RAG.")
        return None
    product_df = dataframe[dataframe['product'].str.lower() == product_name.lower()].copy()
    if product_df.empty:
        return None
    documents = [Document(page_content=row['text']) for index, row in product_df.iterrows()]
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = Chroma.from_documents(documents, embeddings)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key, temperature=0.5)
    prompt_template = f"""
    Sua missão é atuar como {persona_name}, uma persona sintética que representa um cliente da Nomad que usa o produto '{product_name}'.
    Você NÃO é uma IA, um especialista ou um porta-voz. Você é um cliente comum.
    O CONTEXTO abaixo contém uma coleção de comentários e opiniões REAIS de diferentes pessoas. Sua tarefa é LER e SINTETIZAR essas opiniões para formar uma resposta COESA e NATURAL, como se você fosse uma única pessoa que possui todas essas experiências.
    REGRAS DE ATUAÇÃO (MUITO IMPORTANTES):
    1.  **PERSONA E TOM:** Responda em primeira pessoa ("Eu acho", "Na minha experiência..."). Use um tom coloquial, como se estivesse conversando com um amigo. Use interjeições como "Olha...", "Poxa...", "Então...", "né?". Sua resposta deve parecer um áudio de WhatsApp transcrito.
    2.  **SÍNTESE, NÃO LISTAGEM:** NUNCA liste os comentários ou diga "um usuário disse isso...". Integre as ideias em uma única narrativa. Se o contexto tiver opiniões conflitantes (ex: um ama as taxas, outro odeia), sua resposta DEVE refletir essa ambiguidade. Diga algo como: "Olha, sobre as taxas é meio polêmico, né? Já vi gente falando bem, mas eu pessoalmente acho que poderiam melhorar."
    3.  **100% FIEL AO CONTEXTO:** Sua única fonte da verdade é o CONTEXTO abaixo. NÃO invente informações, detalhes, funcionalidades ou experiências que não estejam explicitamente mencionadas. Use os detalhes específicos do contexto (nomes de concorrentes, funcionalidades, lugares) para dar mais vida e credibilidade à sua resposta.
    4.  **SEJA HONESTO SE NÃO SOUBER:** Se o contexto fornecido não contiver NENHUMA informação relevante para responder à pergunta, seja direto e honesto. Diga algo como: "Poxa, sobre isso especificamente eu não tenho experiência pra contar" ou "Ninguém comentou sobre isso, então fico te devendo essa informação". Não tente adaptar um contexto irrelevante.
    ---
    CONTEXTO (Comentários e Opiniões Reais de Clientes):
    {{context}}
    ---
    PERGUNTA DO ENTREVISTADOR:
    {{question}}
    Sua Resposta (como {persona_name}):
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True 
    )
    return rag_chain


def generate_suggested_questions(rag_chain, persona_name, product_name):
    prompt = f"""
    Atue como um Pesquisador de UX e Estrategista de Produto sênior. Sua tarefa é criar exatamente 10 perguntas abertas e investigativas para serem feitas à persona sintética "{persona_name}".
    O contexto do produto é: **{product_name}**.
    O objetivo é que estas perguntas ajudem profissionais de Marketing e Produto a descobrirem insights profundos sobre:
    - O perfil e comportamento do cliente DENTRO DO CONTEXTO DESTE PRODUTO.
    - Suas reais necessidades e dores latentes relacionadas a este produto.
    - Suas motivações e barreiras para usar o produto.
    - Seu nível de conhecimento sobre o tema específico do produto.
    Diretrizes para as perguntas:
    - Crie perguntas que incentivem respostas descritivas, não apenas 'sim' ou 'não'.
    - Use inícios como "O que passa na sua cabeça quando...", "Descreva sua maior frustração com...", "Como você compara...", "Qual a coisa mais importante para você em...".
    - O tom deve ser curioso e empático.
    Retorne o resultado como uma lista de EXATAMENTE 10 strings em Python. Exemplo: ["Pergunta 1?", "Pergunta 2?", ...]
    """
    try:
        response = rag_chain.combine_documents_chain.llm_chain.llm.invoke(prompt)
        suggested_list = eval(response.content)
        if isinstance(suggested_list, list) and len(suggested_list) > 0:
            return suggested_list
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao gerar perguntas sugeridas com IA. Usando lista de fallback. Erro: {e}")
        pass
    fallback_questions = {
        "Conta Internacional": [
            "Qual foi o principal motivo que te fez buscar uma conta em dólar?",
            "Descreva a sua maior dificuldade ao usar seu dinheiro em viagens internacionais.",
            "O que você mais valoriza em um cartão de viagem: taxas baixas, facilidade de uso ou segurança?",
            "Como você se planeja financeiramente para uma viagem ao exterior?",
            "Se você pudesse adicionar uma funcionalidade à conta, qual seria?",
            "Como você compara a Nomad com outras soluções que já usou para viajar?",
            "Qual a sua maior preocupação ao usar um cartão novo em outro país?",
            "O que você gostaria de saber sobre as taxas que ainda não está claro?",
            "Descreva um momento ou situação em que a conta realmente te ajudou ou te surpreendeu.",
            "Que conselho você daria para alguém que vai fazer sua primeira viagem internacional?"
        ],
        "Investimentos no Exterior": [
            "O que te motivou a começar a investir seu dinheiro fora do Brasil?",
            "Qual é a sua maior preocupação ou medo ao pensar em investimentos no exterior?",
            "Descreva como você se sente em relação à volatilidade do mercado de ações americano.",
            "O que você considera mais importante: a possibilidade de altos retornos ou a segurança dos seus investimentos?",
            "Como você avalia seu próprio nível de conhecimento sobre ETFs, Ações e Renda Fixa?",
            "Se você pudesse ter uma informação ou ferramenta a mais para te ajudar a investir, qual seria?",
            "Como você compara investir pela Nomad com outras corretoras que conhece?",
            "Qual é a sua maior dificuldade na hora de declarar seus investimentos no Imposto de Renda?",
            "Descreva o que te dá mais confiança na hora de escolher um ativo para investir.",
            "Que conselho você daria para um amigo que está pensando em começar a investir no exterior?"
        ],
        "App": [
            "Qual foi a primeira coisa que você tentou fazer ao abrir o app pela primeira vez?",
            "Descreva a sua maior frustração ou dificuldade ao usar o aplicativo no dia a dia.",
            "O que você acha mais fácil e mais difícil de encontrar dentro do app?",
            "Se você pudesse mudar uma tela ou um fluxo no aplicativo, qual seria e por quê?",
            "Existe alguma funcionalidade que você esperava encontrar no app e não achou?",
            "Como você descreveria a aparência e a sensação de usar o app para um amigo?",
            "Com que frequência você abre o aplicativo? O que te leva a abri-lo?",
            "Você já enfrentou algum bug ou lentidão? Como foi a experiência?",
            "O que te dá mais segurança ao realizar uma operação financeira pelo app?",
            "Na sua opinião, qual é a funcionalidade mais útil do aplicativo hoje?"
        ]
    }
    return fallback_questions.get(product_name, fallback_questions["App"])
