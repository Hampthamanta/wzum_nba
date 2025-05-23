import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import argparse
from pathlib import Path
import random
import pickle
import time

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
from sklearn.ensemble import VotingClassifier



from prepare_datesets import calculate_correlations
from nba_utilities import show_played_games, show_heatmap




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




def perpare_and_train(df_, who = 'ALLNBA', test_season=2023):
    if who == 'ALLNBA':
        df_ = df_[df_['GP'] >= 40]
    else:
        df_ = df_[df_['GP'] >= 20]

    target='result_top_five'
    t_pname = 'PLAYER_NAME'

    useless_to_model = [col for col in df_.columns if col.endswith('_RANK')] + [
        'PLAYER_ID', 'NICKNAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
    ]
    df_data = df_.drop(useless_to_model, axis=1)

    # Zamiana danych float na int
    float_cols = [col for col in df_data.select_dtypes(include=['float']).columns if col != t_pname]
    df_data[float_cols] = df_data[float_cols].astype(int)

    df_train, df_test = split_train_test(df_data, test_season=test_season)
    
    player_names = np.array(df_test[t_pname])
    X_train = np.array(df_train.drop(columns=[target, t_pname]))
    y_train = np.array(df_train[target])
    X_test  = np.array(df_test.drop(columns=[target, t_pname]))
    y_test  = np.array(df_test[target])

    # Skalowanie danych
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    

    start_time = time.time()

    # Lista modeli decyzyjnych
    models = []

    for k in range(3):
        RS = random.randint(1, 100)
        n_estimators = random.randint(50,500)

        # models.append(RandomForestClassifier(random_state=RS, n_estimators=n_estimators))
        models.append(LogisticRegression(random_state=RS, solver='lbfgs', max_iter=1000))
        models.append(KNeighborsClassifier(n_neighbors=12, weights='distance', metric='euclidean'))
        # models.append(KNeighborsClassifier(n_neighbors=12, weights='distance', metric='manhattan'))
        # models.append(MLPClassifier(hidden_layer_sizes=(120, 60, 30), activation='relu', solver='lbfgs', max_iter=1000, random_state=RS))
        models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.001, max_iter=1000, random_state=RS, early_stopping=True))
        models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', max_iter=1000, random_state=RS))
        models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=0.95))
        models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=1.0))
        # models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=1.1))
        # models.append(GradientBoostingClassifier(random_state=RS, n_estimators=n_estimators))


    for k in range(5):
        RS = random.randint(1, 100)
        n_estimators = random.randint(50,500)
        # models.append(LogisticRegression(random_state=RS, solver='lbfgs', max_iter=1000))
        models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.001, max_iter=1000, random_state=RS, early_stopping=True))
        models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', max_iter=1000, random_state=RS))
        models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=1.0))
        
        
    # for k in range(10):
    #     RS = random.randint(1, 100)
    #     n_estimators = random.randint(80,800)
    #     models.append(RandomForestClassifier(random_state=RS, n_estimators=n_estimators))


    
    votes_for_first_team = {}
    votes_for_second_team = {}
    votes_for_third_team = {}

    for k, model in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for pred, player_name in zip(y_pred, player_names):
            if pred == 1:
                votes_for_first_team[player_name] = votes_for_first_team.get(player_name,0) + 1
            elif pred == 2:
                votes_for_second_team[player_name] = votes_for_second_team.get(player_name,0) + 1
            elif pred == 3:
                votes_for_third_team[player_name] = votes_for_third_team.get(player_name,0) + 1


    # sortowanie głosów
    votes_for_first_team = sorted(votes_for_first_team.items(), key=lambda x: x[1], reverse=True)
    votes_for_second_team = sorted(votes_for_second_team.items(), key=lambda x: x[1], reverse=True)
    votes_for_third_team = sorted(votes_for_third_team.items(), key=lambda x: x[1], reverse=True)

    # decydowanie / odrzucanie duplikatów
    first_team = []
    second_team = []
    third_team = []
    players_in_team = set()

    for name, votes in votes_for_first_team:
        if name not in players_in_team and len(first_team) < 5:
            first_team.append(name)
            players_in_team.add(name)

    for name, votes in votes_for_second_team:
        if name not in players_in_team and len(second_team) < 5:
            second_team.append(name)
            players_in_team.add(name)
 
    for name, votes in votes_for_third_team:
        if name not in players_in_team and len(third_team) < 5:
            third_team.append(name)
            players_in_team.add(name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f'Czas wykonania: {minutes:.2f} min {seconds:.2f} s')

    print("First team:", first_team)
    print("Second team:", second_team)
    print("Third team:", third_team)

    ################## WERYFIKACJA ##################

    print('=='*20)
    print('Real votes for First team:')
    real_votes_for_first_team = df_test[df_test[target] == 1][t_pname].values
    real_votes_for_second_team = df_test[df_test[target] == 2][t_pname].values
    real_votes_for_third_team = df_test[df_test[target] == 3][t_pname].values

    check_first_team = set(first_team) & set(real_votes_for_first_team)
    check_second_team = set(second_team) & set(real_votes_for_second_team)
    check_third_team = set(third_team) & set(real_votes_for_third_team)
    
    print(f'Pierwsza piatka zgadniętych = {len(check_first_team)}')
    print(f'Druga piatka zgadniętych = {len(check_second_team)}')
    print(f'Trzecia piatka zgadniętych = {len(check_third_team)}')


    ################## CZY ZAPISAĆ MODEL ##################
    save_model = input('Czy zapisać model? (y/n): ')
    if save_model == 'y':
        with open(f'models/models_{time.strftime("%Y-%m-%d")}.pkl', 'wb') as f:
            pickle.dump(models, f)




def main():
    df_rookie = pd.read_csv('nba_basic_stats_rookie.csv')
    df_allnba = pd.read_csv('nba_basic_stats_allnba.csv')

    print('ALLNBA')
    perpare_and_train(df_=df_allnba, who='ALLNBA', test_season=2022)
    print('\n\nROOKIE')
    # perpare_and_train(df_=df_rookie, who='ROOKIE', test_season=2023)

    # show_played_games(df_allnba)



if __name__ == "__main__":
    main()
