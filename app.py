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
    st.session_state.messages = []
    st.session_state.question_count = 0
    st.session_state.rag_chain = None
    st.session_state.persona_name = ""
    st.session_state.suggested_questions = []

# --- Função para o Footer ---
def render_footer():
    st.markdown("---")
    st.markdown("Desenvolvido por [Pedro Costa](https://www.linkedin.com/in/pedrocsta/) | Product Marketing & Martech Specialist")

# =============================================================================
# TELA 1: HOME / SELEÇÃO
# =============================================================================
def render_home_screen():
    st.title("Data Persona Interativa: O Diálogo Direto com Seus Dados")

    st.markdown("""
    Esta aplicação cria uma persona interativa e 100% data-driven, utilizando a arquitetura **RAG (Retrieval-Augmented Generation)** e um modelo de linguagem avançado. Diferente de um chatbot, ela responde exclusivamente com base no conhecimento que você fornece (pesquisas, social listening, reviews), garantindo insights autênticos e focados.

    Seu verdadeiro poder é a **autonomia**. Em vez de iniciar um novo ciclo de análise para cada pergunta, a ferramenta transforma seus dados estáticos em um **ativo conversacional**. Explore os resultados de suas pesquisas ou os comentários de redes sociais usando linguagem natural, a qualquer hora.

    É o Martech aplicado na prática: um recurso para que times de Marketing e Produto validem premissas e aprofundem a empatia com o cliente de forma ágil e sem intermediários.
    """)

    with st.expander("⚙️ Conheça o maquinário por trás da mágica"):
        st.markdown("""
        - **Modelo de Linguagem (LLM):** `Google Gemini 1.5 Pro`
        - **Arquitetura:** `RAG (Retrieval-Augmented Generation)`
        - **Orquestração:** `LangChain`
        - **Interface e Aplicação:** `Python + Streamlit`
        - **Base de Dados Vetorial:** `ChromaDB (in-memory)`
        """)

    st.divider()

    st.selectbox(
        'Selecione a Marca:',
        ('Nomad',),
        help="Para esta versão Beta, apenas a marca Nomad está disponível."
    )
    st.caption("Em breve: Integração com Wise e Avenue.")

    selected_brand = "Nomad"

    selected_product = st.selectbox(
        'Selecione o Produto para a Persona:',
        ('Conta Internacional', 'Investimentos no Exterior')
    )

    if st.button("Iniciar Entrevista", type="primary"):
        if "GEMINI_API_KEY" not in st.secrets:
            st.error("Chave da API não configurada. O admin do app precisa adicioná-la nos segredos do Streamlit Cloud.")
            st.stop()

        api_key = st.secrets["GEMINI_API_KEY"]

        with st.spinner("Preparando a persona... Isso pode levar um momento."):
            full_data = load_and_preprocess_data("data")

            if full_data.empty:
                st.error("Nenhum dado válido encontrado na pasta 'data'. Verifique se existem arquivos .csv com as colunas 'text' e 'product'.")
                st.stop()

            st.session_state.persona_name = PERSONA_NAMES[selected_product]
            rag_chain = create_rag_chain(full_data, selected_product, st.session_state.persona_name, api_key)

            if rag_chain is None:
                st.error(f"Não foram encontrados dados para o produto '{selected_product}'. Verifique seus arquivos CSV.")
            else:
                st.session_state.rag_chain = rag_chain
                # MUDANÇA AQUI: Passando o 'selected_product' para a função
                st.session_state.suggested_questions = generate_suggested_questions(rag_chain, st.session_state.persona_name, selected_product)
                st.session_state.screen = 'chat'
                st.session_state.messages = []
                st.session_state.question_count = 0
                st.rerun()

    render_footer()

# =============================================================================
# TELA 2 e 3: CHAT
# =============================================================================
def render_chat_screen():
    st.title(f"Entrevistando: {st.session_state.persona_name}")
    st.markdown(f"Você pode fazer até **{5 - st.session_state.question_count}** pergunta(s).")
    st.divider()

    col1, col2 = st.columns([3, 1])

    with col1:
        if 'messages' in st.session_state:
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
            st.warning("Você atingiu o limite de 5 perguntas para esta sessão. Inicie uma nova entrevista.")

    with col2:
        with st.container(border=True):
            st.subheader("Tópicos sugeridos:")
            if 'suggested_questions' in st.session_state and st.session_state.suggested_questions:
                for i, question in enumerate(st.session_state.suggested_questions):
                    if st.button(question, use_container_width=True, key=f"suggestion_{i}"):
                        st.session_state.messages.append({"role": "user", "content": question})
                        with st.chat_message("user"):
                            st.markdown(question)

                        with st.chat_message("assistant"):
                            with st.spinner("Pensando..."):
                                response = st.session_state.rag_chain.invoke(question)
                                st.markdown(response['result'])

                        st.session_state.messages.append({"role": "assistant", "content": response['result']})
                        st.session_state.question_count += 1
                        st.rerun()

    if st.button("⬅️ Iniciar Nova Entrevista"):
        st.session_state.screen = 'home'
        for key in list(st.session_state.keys()):
            if key != 'screen':
                del st.session_state[key]
        st.rerun()

    render_footer()

# --- Lógica Principal para Alternar Telas ---
if 'screen' not in st.session_state:
    st.session_state.screen = 'home'

if st.session_state.screen == 'home':
    render_home_screen()
elif st.session_state.screen == 'chat':
    render_chat_screen()
