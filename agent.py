%%bash
cat > /content/project/agent.py <<'PY'
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eda import summary_stats, hist_plot, corr_matrix, detect_outliers_isolationforest
from sklearn.cluster import KMeans

# Novo: usa OpenAI opcionalmente
import openai

class Agent:
    def __init__(self, df, memory):
        self.df = df.copy()
        self.memory = memory
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()
        self.mode = os.getenv("AGENT_MODE", "rule")
        if self.mode == "llm":
            openai.api_key = os.getenv("OPENAI_API_KEY")

    # Função principal
    def handle_question(self, question: str):
        if self.mode == "llm":
            return self._handle_llm(question)
        else:
            return self._handle_rule(question)

    # --- 1️⃣ MODO LLM: agente automático e interpretativo ---
    def _handle_llm(self, question: str):
        """
        O agente pergunta ao modelo OpenAI qual tipo de análise deve executar
        e depois executa a função correspondente automaticamente.
        """
        try:
            columns = list(self.df.columns)
            sample = self.df.head(3).to_dict()
            context = f"O dataset contém {len(columns)} colunas: {columns}. Exemplo de dados: {sample}."

            prompt = f"""
Você é um analista de dados especializado em EDA (Exploração de Dados) e Pandas.
Dado o dataset abaixo, o usuário fez a pergunta:

Pergunta: "{question}"

{context}

Sua tarefa:
1. Identifique o tipo de análise (entre: estatísticas, histograma, correlação, outliers, clusters, frequências, tendência temporal, descrição geral).
2. Gere uma explicação curta e clara em português.
3. Se possível, indique qual coluna usar.

Responda em JSON com os campos:
- action: tipo da ação
- column: nome da coluna se aplicável
- answer: explicação textual da resposta
"""
            res = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=500
            )
            import json
            response = json.loads(res.choices[0].message.content)

            action = response.get("action","")
            col = response.get("column")
            text = response.get("answer","")

            # Executa ação correspondente
            result = {"text": text}
            if action in ["estatísticas","summary","stats"]:
                table = summary_stats(self.df)
                result["table"] = table
            elif action in ["histograma","distribuição","hist"]:
                if col not in self.df.columns:
                    col = self.numeric_cols[0]
                fig = hist_plot(self.df, col)
                result["fig"] = fig
            elif action in ["correlação","relacionamento"]:
                table = corr_matrix(self.df)
                result["table"] = table
            elif action in ["outliers","anomalias"]:
                out = detect_outliers_isolationforest(self.df, self.numeric_cols)
                result["table"] = out.head(50)
            elif action in ["clusters","agrupamento"]:
                km = KMeans(n_clusters=3, random_state=42).fit(self.df[self.numeric_cols].fillna(0))
                df2 = self.df.copy()
                df2["_cluster"] = km.labels_
                result["table"] = df2["_cluster"].value_counts().reset_index().rename(columns={'index':'cluster','_cluster':'count'})
            elif action in ["frequências","moda"]:
                freq = {col: self.df[col].value_counts().head(5).to_dict() for col in self.categorical_cols[:5]}
                result["table"] = pd.DataFrame(freq)
            elif action in ["tendência","temporal"]:
                time_cols = [c for c in self.df.columns if "time" in c.lower() or "date" in c.lower()]
                if time_cols:
                    col = time_cols[0]
                    ts = self.df.groupby(col).size()
                    fig, ax = plt.subplots()
                    ts.plot(ax=ax)
                    ax.set_title(f"Tendência temporal por {col}")
                    result["fig"] = fig

            self.memory.add_interaction(question, text, meta={"mode":"llm","action":action})
            return result

        except Exception as e:
            fallback = f"Erro no modo LLM: {e}. Tentando modo rule."
            self.memory.add_interaction(question, fallback)
            return self._handle_rule(question)

    # --- 2️⃣ MODO RULE: fallback offline (sem API) ---
    def _handle_rule(self, question: str):
        q = question.lower()
        if "tipo" in q:
            text = f"Colunas numéricas: {self.numeric_cols}\nColunas categóricas: {self.categorical_cols}"
            self.memory.add_interaction(question, text)
            return {"text": text}
        if "distribui" in q or "histograma" in q:
            col = self.numeric_cols[0]
            fig = hist_plot(self.df, col)
            text = f"Histograma gerado para {col}."
            self.memory.add_interaction(question, text)
            return {"text": text, "fig": fig}
        if "correlação" in q:
            table = corr_matrix(self.df)
            self.memory.add_interaction(question, "Matriz de correlação calculada.")
            return {"text": "Matriz de correlação calculada.", "table": table}
        if "outlier" in q or "anomalia" in q:
            out = detect_outliers_isolationforest(self.df, self.numeric_cols)
            text = f"Foram detectados {len(out)} outliers."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": out.head(20)}
        text = "Modo rule: não entendi. Tente 'tipos de dados', 'histograma', 'correlação', 'outliers'."
        self.memory.add_interaction(question, text)
        return {"text": text}
PY
