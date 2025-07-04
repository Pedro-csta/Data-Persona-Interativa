# app.py
import streamlit as st
from utils import PERSONA_NAMES
from rag_components import load_and_preprocess_data, create_rag_chain, generate_suggested_questions

# --- Configuração da Página ---
st.set_page_config(
    page_title="Data Persona Interativa - Nomad",
    page_icon="🤖",
    layout="wide"
)

# --- Gerenciamento de Estado da Sessão ---
if 'screen' not in st.session_state:
    st.session_state.screen = 'home'
    # ... (inicialização de outros estados, se necessário)

# =============================================================================
# TELA 1: HOME / SELEÇÃO
# =============================================================================
def render_home_screen():
    st.title("Pesquisa / Entrevista com uma Data Persona")
    st.markdown("""
    Bem-vindo à ferramenta de **Data Persona Interativa**. Esta aplicação utiliza um modelo de linguagem avançado 
    (Google Gemini) sob a arquitetura **RAG (Retrieval-Augmented Generation)**. 

    Isso significa que a persona com quem você irá conversar não usa o conhecimento geral da internet, 
    mas sim uma **base de dados real e exclusiva** sobre clientes da marca, garantindo insights autênticos e focados.
    """)
    st.divider()

    selected_brand = st.selectbox(
        'Selecione a Marca:',
        ('Nomad', 'Wise (em breve)', 'Avenue (em breve)'),
        disabled=(False, True, True)
    )

    selected_product = st.selectbox(
        'Selecione o Produto para a Persona:',
        ('Conta Internacional', 'Investimentos no Exterior')
    )

    if st.button("Iniciar Entrevista", type="primary"):
        # **MUDANÇA IMPORTANTE AQUI**
        # Busca a chave da API dos segredos do Streamlit
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Chave da API não configurada. O admin do app precisa adicioná-la nos segredos do Streamlit Cloud.")
            st.stop()

        api_key = st.secrets["GEMINI_API_KEY"]

        with st.spinner("Preparando a persona... Isso pode levar um momento."):
            full_data = load_and_preprocess_data("data/knowledge_base_nomad.csv")
            st.session_state.persona_name = PERSONA_NAMES[selected_product]
            rag_chain = create_rag_chain(full_data, selected_product, st.session_state.persona_name, api_key)

            if rag_chain is None:
                st.error(f"Não foram encontrados dados para o produto '{selected_product}'. Verifique seu arquivo CSV.")
            else:
                st.session_state.rag_chain = rag_chain
                st.session_state.suggested_questions = generate_suggested_questions(rag_chain, st.session_state.persona_name)
                st.session_state.screen = 'chat'
                st.session_state.messages = []
                st.session_state.question_count = 0
                st.rerun()

# =============================================================================
# TELA 2 e 3: CHAT
# =============================================================================
def render_chat_screen():
    st.title(f"Entrevistando: {st.session_state.persona_name}")
    st.markdown(f"Você pode fazer até **{5 - st.session_state.question_count}** pergunta(s).")
    st.divider()

    col1, col2 = st.columns([3, 1])

    with col1:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.question_count < 5:
            if prompt := st.chat_input("Digite para conversar!"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Pensando..."):
                        response = st.session_state.rag_chain.invoke(prompt)
                        st.markdown(response['result'])

                st.session_state.messages.append({"role": "assistant", "content": response['result']})
                st.session_state.question_count += 1
                st.rerun()
        else:
            st.warning("Você atingiu o limite de 5 perguntas. Inicie uma nova entrevista.")

    with col2:
        with st.container(border=True):
            st.subheader("Tópicos sugeridos:")
            if 'suggested_questions' in st.session_state and st.session_state.suggested_questions:
                for question in st.session_state.suggested_questions:
                    if st.button(question, use_container_width=True, key=question):
                        st.session_state.messages.append({"role": "user", "content": question})
                        # ... (resto da lógica do botão)
                        st.rerun()

    if st.button("⬅️ Iniciar Nova Entrevista"):
        st.session_state.screen = 'home'
        st.rerun()

# --- Lógica Principal ---
if st.session_state.screen == 'home':
    render_home_screen()
elif st.session_state.screen == 'chat':
    render_chat_screen()
