import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def show_played_games(df: pd.DataFrame):
    df_top5 = df[df['result_top_five'] > 0]
    min_gp = df_top5['GP'].min()
    print(f'Min GP: {min_gp}')

    seasons = sorted(df['SEASON'].unique())

    for season in seasons:
        df_season = df[(df['SEASON'] == season) & (df['result_top_five'] > 0)]

        plt.figure(figsize=(8, 5))
        plt.hist(df_season['GP'], bins=range(df_season['GP'].min(), df_season['GP'].max() + 2, 2), edgecolor='black')
        plt.xlabel('Liczba rozegranych gier (GP)')
        plt.ylabel('Liczba zawodników')
        plt.title(f'Rozkład GP dla piątek, sezon {season}')
        plt.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        plt.show()


def show_heatmap(y_test: np.ndarray, y_pred: np.ndarray):
    labels_to_show = [1, 2, 3]
    cm_digits = confusion_matrix(y_test, y_pred, labels=labels_to_show)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_digits, display_labels=labels_to_show)
    disp.plot(cmap="cividis")
    plt.show()