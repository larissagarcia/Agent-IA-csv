import streamlit as st
import pandas as pd
from memory import Memory
from agent import Agent
from report_generator import generate_pdf_report
import os

# Ativa o modo LLM (inteligente)
os.environ.setdefault("AGENT_MODE", "llm")

# Configure a chave de API (no Colab use getpass(), no Cloud use secrets)
if not os.getenv("OPENAI_API_KEY"):
    from getpass import getpass
    os.environ["OPENAI_API_KEY"] = getpass("Digite sua chave OpenAI: ")

os.environ["AGENT_MODE"] = "rule"

import streamlit as st

st.set_page_config(page_title="Agente de Análise Inteligente", layout="wide")
st.title("🤖 Agente Inteligente de Exploração de Dados (EDA)")
st.markdown("Faça upload de um arquivo CSV e faça perguntas em linguagem natural sobre os dados.")

uploaded_file = st.file_uploader("📂 Envie um arquivo CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
else:
    st.info("Envie um CSV para iniciar a análise.")

# lê CSV
import io
df = pd.read_csv(io.BytesIO(uploaded.read()))
st.write(f"Linhas: {df.shape[0]} — Colunas: {df.shape[1]}")
st.dataframe(df.head())

# init
mem = Memory("/content/project/memory.sqlite")
agent = Agent(df, mem)

q = st.text_input("Pergunta sobre o dataset (ex.: 'Mostre histograma da coluna Amount')")
if question:
    with st.spinner("Analisando..."):
        result = agent.handle_question(question)

    st.markdown(f"### 💡 Resposta")
    st.write(result.get("text", ""))

    # Se houver tabela
    if "table" in result:
        st.dataframe(result["table"])

    # Se houver figura (gráfico)
    if "fig" in result:
        st.pyplot(result["fig"])

    st.success("Análise concluída!")

if st.button("📑 Gerar Relatório PDF"):
    pdf_path = generate_pdf_report(mem, output_path="Relatorio_Agente_IA.pdf")
    with open(pdf_path, "rb") as f:
        st.download_button("Baixar Relatório", f, file_name="Relatorio_Agente_IA.pdf")
