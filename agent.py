import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eda import summary_stats, hist_plot, corr_matrix, detect_outliers_isolationforest
from sklearn.cluster import KMeans
from openai import OpenAI

class Agent:
    def __init__(self, df, memory):
        self.df = df.copy()
        self.memory = memory
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()
        self.mode = os.getenv("AGENT_MODE", "rule")

        if self.mode == "llm":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # -------------------------------------------------
    # Função principal
    # -------------------------------------------------
    def handle_question(self, question: str):
        if self.mode == "llm":
            return self._handle_llm(question)
        else:
            return self._handle_rule(question)

    # -------------------------------------------------
    # 🔹 MODO LLM (GPT-4o-mini)
    # -------------------------------------------------
    def _handle_llm(self, question: str):
        try:
            columns = list(self.df.columns)
            sample = self.df.head(3).to_dict()
            context = f"O dataset contém {len(columns)} colunas: {columns}. Exemplo: {sample}."

            prompt = f"""
Você é um analista de dados que utiliza pandas e matplotlib.
O usuário forneceu um dataset e fez a seguinte pergunta:

Pergunta: {question}

{context}

Determine:
1. O tipo de análise necessária (entre: estatísticas, histograma, correlação, outliers, clusters, frequências, tendência temporal).
2. Gere uma explicação curta e clara.
Responda em formato JSON com os campos:
{{ "action": "...", "column": "...", "answer": "..." }}
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            import json
            message = response.choices[0].message.content
            result_json = json.loads(message)

            action = result_json.get("action", "")
            col = result_json.get("column")
            text = result_json.get("answer", "")

            result = {"text": text}

            # -------------------------
            # Executa a análise pedida
            # -------------------------
            if action in ["estatísticas", "summary", "stats"]:
                result["table"] = summary_stats(self.df)

            elif action in ["histograma", "distribuição", "hist"]:
                col = col if col in self.df.columns else self.numeric_cols[0]
                result["fig"] = hist_plot(self.df, col)

            elif action in ["correlação", "relacionamento"]:
                result["table"] = corr_matrix(self.df)

            elif action in ["outliers", "anomalias"]:
                result["table"] = detect_outliers_isolationforest(self.df, self.numeric_cols).head(50)

            elif action in ["clusters", "agrupamento"]:
                km = KMeans(n_clusters=3, random_state=42).fit(self.df[self.numeric_cols].fillna(0))
                df2 = self.df.copy()
                df2["_cluster"] = km.labels_
                result["table"] = (
                    df2["_cluster"].value_counts()
                    .reset_index()
                    .rename(columns={"index": "cluster", "_cluster": "count"})
                )

            elif action in ["frequências", "moda"]:
                freq = {c: self.df[c].value_counts().head(5).to_dict() for c in self.categorical_cols[:5]}
                result["table"] = pd.DataFrame(freq)

            elif action in ["tendência", "temporal"]:
                time_cols = [c for c in self.df.columns if "time" in c.lower() or "date" in c.lower()]
                if time_cols:
                    col = time_cols[0]
                    ts = self.df.groupby(col).size()
                    fig, ax = plt.subplots()
                    ts.plot(ax=ax)
                    ax.set_title(f"Tendência temporal por {col}")
                    result["fig"] = fig
                else:
                    result["text"] += "\nNenhuma coluna temporal identificada."

            else:
                result["text"] += "\n(O modelo não especificou claramente o tipo de análise.)"

            # Salva na memória
            self.memory.add_interaction(question, text, meta={"mode": "llm", "action": action})
            return result

        except Exception as e:
            fallback = f"Erro ao usar o modelo LLM: {e}. Tentando modo simples."
            self.memory.add_interaction(question, fallback)
            return self._handle_rule(question)

    # -------------------------------------------------
    # 🔸 MODO RULE (offline, sem OpenAI)
    # -------------------------------------------------
    def _handle_rule(self, question: str):
        q = question.lower()
        if "tipo" in q:
            text = f"Numéricas: {self.numeric_cols}\nCategóricas: {self.categorical_cols}"
            self.memory.add_interaction(question, text)
            return {"text": text}

        if "hist" in q:
            col = self.numeric_cols[0]
            fig = hist_plot(self.df, col)
            text = f"Histograma de {col}."
            self.memory.add_interaction(question, text)
            return {"text": text, "fig": fig}

        if "correl" in q:
            result = corr_matrix(self.df)
            text = "Matriz de correlação calculada."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": result}

        if "outlier" in q or "anomalia" in q:
            result = detect_outliers_isolationforest(self.df, self.numeric_cols)
            text = f"{len(result)} outliers detectados."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": result.head(20)}

        text = "Modo simples: não entendi. Tente 'tipos de dados', 'histograma', 'correlação', 'outliers'."
        self.memory.add_interaction(question, text)
        return {"text": text}
