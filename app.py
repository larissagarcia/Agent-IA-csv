import streamlit as st
import pandas as pd
from memory import Memory
from agent import Agent
from report_generator import generate_pdf_report
import os

# --- Configuração ---
os.environ.setdefault("AGENT_MODE", "llm")

# Solicita chave OpenAI se não estiver definida (no Cloud use Secrets)
if not os.getenv("OPENAI_API_KEY"):
    from getpass import getpass
    os.environ["OPENAI_API_KEY"] = getpass("Digite sua chave OpenAI: ")

# --- Interface ---
st.set_page_config(page_title="Agente Inteligente de Exploração de Dados", layout="wide")
st.title("🤖 Agente Inteligente de Exploração de Dados (EDA)")
st.markdown("Envie um arquivo CSV e faça perguntas em linguagem natural sobre os dados.")

uploaded_file = st.file_uploader("📂 Envie um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(f"**Linhas:** {df.shape[0]} — **Colunas:** {df.shape[1]}")
    st.dataframe(df.head())

    mem = Memory("memory.sqlite")
    agent = Agent(df, mem)

    q = st.text_input("💬 Pergunta sobre o dataset (ex.: 'Quais variáveis são mais correlacionadas?')")
    if q:
        with st.spinner("Analisando..."):
            result = agent.handle_question(q)

        st.markdown("### 💡 Resposta")
        st.write(result.get("text", ""))

        if "table" in result:
            st.dataframe(result["table"])
        if "fig" in result:
            st.pyplot(result["fig"])

        st.success("✅ Análise concluída!")

    if st.button("📑 Gerar Relatório PDF"):
        pdf_path = generate_pdf_report(mem, output_path="Relatorio_Agente_IA.pdf")
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Baixar Relatório", f, file_name="Relatorio_Agente_IA.pdf")

else:
    st.info("Envie um CSV para iniciar a análise.")
