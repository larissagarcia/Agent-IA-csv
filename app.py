import streamlit as st
import pandas as pd
from memory import Memory
from agent import Agent
from report_generator import generate_pdf_report

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
if st.button("Enviar pergunta"):
    if not q.strip():
        st.warning("Digite uma pergunta.")
    else:
        with st.spinner("Processando..."):
            resp = agent.handle_question(q)
        st.markdown("### Resposta do agente")
        st.write(resp.get("text"))
        if "table" in resp:
            st.dataframe(resp["table"])
        if "fig" in resp:
            st.pyplot(resp["fig"])

if st.button("Gerar relatório PDF"):
    pdf_path = generate_pdf_report(mem, output_path="/content/project/Agentes_Autonomos_Relatorio.pdf")
    with open(pdf_path, "rb") as f:
        st.download_button("Download Relatório PDF", f, file_name="Agentes_Autonomos_Relatorio.pdf")
