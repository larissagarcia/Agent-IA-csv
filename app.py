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

st.set_page_config(page_title="Agente EDA (Colab)", layout="wide")
st.title("Agente E.D.A. — Colab")

uploaded = st.file_uploader("Carregue um CSV", type=["csv","zip"])
if uploaded is None:
    st.info("Envie um CSV para começar (ex.: creditcardfraud.csv).")
    st.stop()

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

if st.button("Gerar relatório PDF"):
    pdf_path = generate_pdf_report(mem, output_path="Agentes_Autonomos_Relatorio.pdf")
    with open(pdf_path, "rb") as f:
        st.download_button("Download Relatório PDF", f, file_name="Agentes_Autonomos_Relatorio.pdf")
