import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def summary_stats(df):
    return df.describe().T

def hist_plot(df, column):
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=20, edgecolor='black')
    ax.set_title(f'Distribuição de {column}')
    return fig

def corr_matrix(df):
    return df.corr(numeric_only=True)

def detect_outliers_isolationforest(df, numeric_cols):
    iso = IsolationForest(contamination=0.05, random_state=42)
    outliers = df[numeric_cols].copy()
    outliers['outlier'] = iso.fit_predict(outliers)
    return df[outliers['outlier'] == -1]
