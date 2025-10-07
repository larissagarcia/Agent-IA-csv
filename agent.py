import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eda import summary_stats, hist_plot, corr_matrix, detect_outliers_isolationforest
from sklearn.cluster import KMeans
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

    def handle_question(self, question: str):
        if self.mode == "llm":
            return self._handle_llm(question)
        else:
            return self._handle_rule(question)

    # ---------- MODO INTELIGENTE ----------
    def _handle_llm(self, question: str):
        try:
            columns = list(self.df.columns)
            sample = self.df.head(3).to_dict()
            context = f"O dataset contém {len(columns)} colunas: {columns}. Exemplo: {sample}."

            prompt = f"""
Você é um analista de dados que usa pandas e matplotlib.
Com base no dataset descrito abaixo, o usuário perguntou:

Pergunta: {question}

{context}

Determine:
1. Qual tipo de análise deve ser feita (entre: estatísticas, histograma, correlação, outliers, clusters, frequências, tendência temporal).
2. Gere uma resposta curta e explicativa.
Responda em JSON: {{ "action": "...", "column": "...", "answer": "..." }}
"""
            res = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            import json
            response = json.loads(res.choices[0].message.content)

            action = response.get("action", "")
            col = response.get("column")
            text = response.get("answer", "")

            result = {"text": text}

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
                result["table"] = df2["_cluster"].value_counts().reset_index().rename(columns={'index': 'cluster', '_cluster': 'count'})
            elif action in ["frequências", "moda"]:
                freq = {c: self.df[c].value_counts().head(5).to_dict() for c in self.categorical_cols[:5]}
                result["table"] = pd.DataFrame(freq)

            self.memory.add_interaction(question, text, meta={"mode": "llm", "action": action})
            return result

        except Exception as e:
            self.memory.add_interaction(question, f"Erro LLM: {e}")
            return self._handle_rule(question)

    # ---------- MODO SIMPLES ----------
    def _handle_rule(self, question: str):
        q = question.lower()
        if "tipo" in q:
            text = f"Numéricas: {self.numeric_cols}\\nCategóricas: {self.categorical_cols}"
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
        if "outlier" in q:
            result = detect_outliers_isolationforest(self.df, self.numeric_cols)
            text = f"{len(result)} outliers detectados."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": result.head(20)}
        text = "Não entendi. Tente: 'tipos de dados', 'histograma', 'correlação', 'outliers'."
        self.memory.add_interaction(question, text)
        return {"text": text}
