# app.py
import streamlit as st
from utils import PERSONA_NAMES
# A importação agora reflete a nova estrutura de agentes
from rag_components import load_and_preprocess_data, get_retriever, create_agentic_rag_app, generate_suggested_questions

st.set_page_config(page_title="Data Persona Interativa", page_icon="🤖", layout="wide")

# Inicialização do estado da sessão
if 'screen' not in st.session_state:
    st.session_state.screen = 'home'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agentic_app' not in st.session_state:
    st.session_state.agentic_app = None

def render_footer():
    st.markdown("---")
    st.markdown("Desenvolvido por [Pedro Costa](https://www.linkedin.com/in/pedrocsta/) | Product Marketing & Martech Specialist")

def render_home_screen():
    st.title("Data Persona Interativa 💬")
    # ... (o resto do texto de apresentação permanece o mesmo) ...
    st.markdown("""
    Esta aplicação cria uma persona interativa e 100% data-driven...
    Seu verdadeiro poder é a autonomia...
    É o Martech aplicado na prática...
    """)
    with st.expander("⚙️ Conheça o maquinário por trás da mágica"):
        st.markdown("""
        - **Modelo de Linguagem (LLM):** `Google Gemini 1.5 Pro & Flash`
        - **Arquitetura:** `RAG com Agentes de IA (LangGraph)`
        - **Orquestração:** `LangChain`
        - **Interface e Aplicação:** `Python + Streamlit`
        - **Base de Dados Vetorial:** `ChromaDB (in-memory)`
        """)
    st.divider()
    st.selectbox('Selecione a Marca:', ('Nomad',), help="Para esta versão Beta, apenas a marca Nomad está disponível.")
    selected_product = st.selectbox(
        'Selecione o Produto para a Persona:',
        ('Conta Internacional', 'Investimentos no Exterior', 'App')
    )

    if st.button("Iniciar Entrevista", type="primary"):
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Chave da API não configurada.")
            st.stop()
        api_key = st.secrets["GEMINI_API_KEY"]
        
        with st.spinner("Preparando a persona e seus agentes..."):
            full_data = load_and_preprocess_data("data")
            if full_data.empty: st.error("Nenhum dado válido na pasta 'data'."); st.stop()
            
            retriever = get_retriever(full_data, selected_product, api_key)
            if retriever is None: st.error(f"Não há dados para o produto '{selected_product}'."); st.stop()
            
            # Cria e armazena o aplicativo de agentes na sessão
            st.session_state.agentic_app = create_agentic_rag_app(retriever, api_key)
            st.session_state.persona_name = PERSONA_NAMES[selected_product]
            st.session_state.product_name = selected_product # Armazena o produto para o prompt
            st.session_state.suggested_questions = generate_suggested_questions(api_key, st.session_state.persona_name, selected_product)
            
            st.session_state.screen = 'chat'
            st.session_state.messages = []
            st.rerun()
    render_footer()

def handle_new_message(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Prepara o payload para o grafo de agentes
    payload = {
        "question": prompt,
        "chat_history": [(msg["role"], msg["content"]) for msg in st.session_state.messages[:-1]],
        "product_name": st.session_state.product_name,
        "persona_name": st.session_state.persona_name
    }
    
    with st.chat_message("assistant"):
        with st.spinner("A equipe de agentes está pensando..."):
            # Invoca o grafo de agentes
            final_state = st.session_state.agentic_app.invoke(payload)
            response_content = final_state['final_answer']
            st.markdown(response_content)
            # Adiciona as fontes para depuração
            with st.expander("Ver fontes utilizadas"):
                for doc in final_state['documents']:
                    st.info(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": response_content})

def render_chat_screen():
    st.title(f"Entrevistando: {st.session_state.persona_name}")
    st.markdown(f"Você pode fazer até **5** perguntas. Esta é uma demonstração.")
    st.divider()

    # Exibe o histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Lida com o input do usuário
    if len(st.session_state.messages) < 10: # Limite de 5 perguntas (5 pares de user/assistant)
        if prompt := st.chat_input("Digite para conversar!"):
            handle_new_message(prompt)
            st.rerun()
    else:
        st.warning("Você atingiu o limite de perguntas para esta demonstração.")

    if st.button("⬅️ Iniciar Nova Entrevista"):
        keys_to_clear = ['messages', 'agentic_app', 'persona_name', 'product_name', 'suggested_questions']
        for key in keys_to_clear:
            if key in st.session_state: del st.session_state[key]
        st.session_state.screen = 'home'
        st.rerun()
        
    render_footer()

# --- Lógica Principal ---
if st.session_state.screen == 'home':
    render_home_screen()
elif st.session_state.screen == 'chat':
    render_chat_screen()
