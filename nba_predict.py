import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import argparse
from pathlib import Path
import random

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from prepare_datesets import calculate_correlations


# RS = random.randint(1, 100)
RS = 42





def split_train_test(df, test_season=2023):
    # Usuń dane z sezonu 2025 (niekompletne)
    df = df[df['SEASON'] != 2025]

    # Podziel na train/test
    df_train = df[df['SEASON'] != test_season].copy()
    df_test  = df[df['SEASON'] == test_season].copy()

    # Wyrzuć kolumnę SEASON z obu zbiorów (nie używasz jej do trenowania)
    if 'SEASON' in df_train.columns:
        df_train = df_train.drop('SEASON', axis=1)
    if 'SEASON' in df_test.columns:
        df_test = df_test.drop('SEASON', axis=1)

    return df_train, df_test




def perpare_and_train(df_):
    useless_to_model = [col for col in df_.columns if col.endswith('_RANK')] + [
        'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'TEAM_ID',
    ]
    df_data = df_.drop(useless_to_model, axis=1)

    # Zamiana danych float na int
    float_cols = df_data.select_dtypes(include=['float']).columns
    df_data[float_cols] = df_data[float_cols].astype(int)

    df_train, df_test = split_train_test(df_data, test_season=2023)

    target='result_top_five'
    X_train = np.array(df_train.drop(columns=[target]))
    y_train = np.array(df_train[target])
    X_test  = np.array(df_test.drop(columns=[target]))
    y_test  = np.array(df_test[target])

    models = [
        RandomForestClassifier(random_state=RS, n_estimators=100),
        LogisticRegression(random_state=RS, solver='lbfgs'),
        KNeighborsClassifier(random_state=RS, n_neighbors=8, weights="uniform"),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=RS),
        SVC(random_state=RS, gamma=2, C=1),
        GradientBoostingClassifier(random_state=RS, n_estimators=100),
        MLPClassifier(random_state=RS, max_iter=1000),
        AdaBoostClassifier(random_state=RS, n_estimators=300),
    ]




def main():
    df_rookie = pd.read_csv('nba_basic_stats_allnba.csv')
    df_allnba = pd.read_csv('nba_basic_stats_allnba.csv')

    print('ALLNBA')
    perpare_and_train(df_allnba)
    print('\n\nROOKIE')
    perpare_and_train(df_rookie)



if __name__ == "__main__":
    main()
