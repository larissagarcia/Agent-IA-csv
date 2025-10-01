import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eda import summary_stats, hist_plot, corr_matrix, detect_outliers_isolationforest
from sklearn.cluster import KMeans

class Agent:
    def __init__(self, df, memory):
        self.df = df.copy()
        self.memory = memory
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=['number']).columns.tolist()

    def handle_question(self, question: str):
        q = question.lower()

        # --- 1) Descrição dos Dados ---
        if "tipo" in q or "dados" in q:
            text = f"Colunas numéricas: {self.numeric_cols}\nColunas categóricas: {self.categorical_cols}"
            self.memory.add_interaction(question, text)
            return {"text": text}

        if "distribui" in q or "histograma" in q:
            figs = []
            for col in self.numeric_cols[:6]:  # limita para não explodir a tela
                figs.append(hist_plot(self.df, col))
            text = "Distribuição (histogramas) das variáveis numéricas principais."
            self.memory.add_interaction(question, text)
            return {"text": text, "fig": figs[0]}  # mostra o primeiro, mas poderia iterar

        if "intervalo" in q or "mínimo" in q or "máximo" in q:
            stats = self.df[self.numeric_cols].agg(['min','max']).T
            text = "Intervalos (mínimo e máximo) das variáveis numéricas."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": stats}

        if "média" in q or "mediana" in q or "tendência central" in q:
            stats = self.df[self.numeric_cols].agg(['mean','median']).T
            text = "Médias e medianas das variáveis numéricas."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": stats}

        if "variabilidade" in q or "desvio padrão" in q or "variância" in q:
            stats = self.df[self.numeric_cols].agg(['std','var']).T
            text = "Medidas de variabilidade (desvio padrão e variância)."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": stats}

        # --- 2) Padrões e Tendências ---
        if "tendência" in q or "temporal" in q:
            # tenta detectar colunas de tempo
            time_cols = [c for c in self.df.columns if "time" in c.lower() or "date" in c.lower()]
            if time_cols:
                col = time_cols[0]
                ts = self.df.groupby(col).size()
                fig, ax = plt.subplots()
                ts.plot(ax=ax)
                ax.set_title(f"Tendência temporal por {col}")
                text = f"Tendência temporal detectada pela coluna {col}."
                self.memory.add_interaction(question, text)
                return {"text": text, "fig": fig}
            else:
                text = "Não foi encontrada coluna temporal explícita."
                self.memory.add_interaction(question, text)
                return {"text": text}

        if "frequente" in q or "menos frequente" in q or "moda" in q:
            freq = {}
            for col in self.categorical_cols[:5]:
                freq[col] = self.df[col].value_counts().head(5).to_dict()
            text = "Valores mais frequentes por variável categórica."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": pd.DataFrame(freq)}

        if "cluster" in q or "agrup" in q:
            if len(self.numeric_cols) >= 2:
                km = KMeans(n_clusters=3, random_state=42).fit(self.df[self.numeric_cols].fillna(0))
                df2 = self.df.copy()
                df2["_cluster"] = km.labels_
                text = "Clusterização aplicada (k=3)."
                self.memory.add_interaction(question, text)
                return {"text": text, "table": df2["_cluster"].value_counts().reset_index().rename(columns={'index':'cluster','_cluster':'count'})}
            else:
                return {"text":"Poucas colunas numéricas para aplicar clusters."}

        # --- 3) Detecção de Outliers ---
        if "outlier" in q or "atípico" in q or "anomalia" in q:
            out = detect_outliers_isolationforest(self.df, self.numeric_cols)
            text = f"Foram detectados {len(out)} outliers usando IsolationForest."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": out.head(50)}

        # --- 4) Relações entre Variáveis ---
        if "correlação" in q or "relacion" in q:
            corr = corr_matrix(self.df)
            text = "Matriz de correlação entre variáveis numéricas."
            self.memory.add_interaction(question, text)
            return {"text": text, "table": corr}

        if "dispersão" in q or "scatter" in q:
            if len(self.numeric_cols) >= 2:
                x, y = self.numeric_cols[:2]
                fig, ax = plt.subplots()
                ax.scatter(self.df[x], self.df[y], alpha=0.3)
                ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"Dispersão entre {x} e {y}")
                text = f"Gráfico de dispersão entre {x} e {y}."
                self.memory.add_interaction(question, text)
                return {"text": text, "fig": fig}
            else:
                return {"text":"Não há colunas numéricas suficientes para scatter plot."}

        # --- fallback ---
        text = "Não reconheci a pergunta. Tente sobre: tipos de dados, distribuições, médias, variância, tendências, frequências, clusters, outliers, correlação."
        self.memory.add_interaction(question, text)
        return {"text": text}
