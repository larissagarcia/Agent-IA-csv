import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def summary_stats(df):
    return df.describe(include='all', datetime_is_numeric=True).T

def hist_plot(df, column, bins=50):
    fig, ax = plt.subplots()
    df[column].dropna().hist(bins=bins, ax=ax)
    ax.set_title(f"Histograma: {column}")
    return fig

def corr_matrix(df):
    return df.corr()

def detect_outliers_isolationforest(df, numeric_cols, contamination=0.01):
    if not numeric_cols:
        return df.iloc[0:0]
    model = IsolationForest(contamination=contamination, random_state=42)
    X = df[numeric_cols].fillna(0)
    preds = model.fit_predict(X)
    return df[preds == -1]
