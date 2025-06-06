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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier


from nba_utilities import show_played_games, show_heatmap, calculate_correlations, print_models_parameters, calculate_correlations_plot




########################################
########################################
USE_SAVED_MODELS = 1
########################################
########################################







def split_train_test(df, test_season=2025):
    df_train = df[df['SEASON'] != test_season].copy()
    df_test  = df[df['SEASON'] == test_season].copy()

    if 'SEASON' in df_train.columns:
        df_train = df_train.drop('SEASON', axis=1)
    if 'SEASON' in df_test.columns:
        df_test = df_test.drop('SEASON', axis=1)

    return df_train, df_test




def fill_team_with_position_limit(votes_list, df_data, already_chosen, max_f=3, max_c=2, max_g=3):
    team = []
    pos_count = {1: 0, 2: 0, 3: 0}
    for name, _ in votes_list:
        if name in already_chosen:
            continue
        pos = df_data.loc[df_data['PLAYER_NAME'] == name, 'Pos']
        if pos.empty or pd.isna(pos.values[0]):
            continue
        pos = int(pos.values[0])
        if pos_count[pos] < (max_f if pos == 1 else max_c if pos == 2 else max_g):
            team.append(name)
            pos_count[pos] += 1
        if len(team) == 5:
            break
    return team





def perpare_and_train(df_, who = 'ALLNBA', test_season=2025, loaded_models=[]):
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


    if who == 'ROOKIE':
        df_data = df_data.drop(columns=['Pos'])
    else:
        pos_dict = {'SF': 1, 'PF': 1, 'C': 2, 'SG': 3, 'PG': 3}
        df_data['Pos'] = df_data['Pos'].map(pos_dict)


    # if who == 'ALLNBA':
    #     df_data = df_data.drop(columns=['PER','TS%','3PAr','FTr','ORB%','DRB%','TRB%','AST%','STL%','BLK%','TOV%','USG%','OWS','DWS','WS','WS/48','OBPM','DBPM','BPM','VORP'])

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
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    

    start_time = time.time()

    # Lista modeli decyzyjnych
    if len(loaded_models) == 0:
        models = []


        # y_pred_list = []
        # for model in models:
        #     print(f'Training model: {type(model).__name__}')
        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)
        #     y_pred_list.append(y_pred)

        # for y_pred in y_pred_list:
        #     show_heatmap(y_test, y_pred)
        # return None, None, None, None, None



        ####### Odtworzenie najlepszych modeli: #######
        if who == 'ALLNBA':
            models.append(LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
            models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=54, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=88, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(256, 256), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=16, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(512, 512), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=67, early_stopping=True))
            models.append(KNeighborsClassifier(n_neighbors=2, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=4, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'))
            models.append(RandomForestClassifier(n_estimators=109, random_state=27))
            models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=27, early_stopping=False))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=27, n_estimators=109))
            models.append(RandomForestClassifier(n_estimators=317, random_state=20))
            models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=20, early_stopping=False))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=20, n_estimators=317))
            models.append(RandomForestClassifier(n_estimators=345, random_state=6))
            models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=6, early_stopping=False))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=6, n_estimators=345))
        else:
            models.append(RandomForestClassifier(n_estimators=300, random_state=42))
            models.append(LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
            models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=54, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=88, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(256, 256), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=16, early_stopping=True))
            models.append(MLPClassifier(hidden_layer_sizes=(512, 512), activation='tanh', solver='lbfgs', alpha=0.0001, max_iter=1000, random_state=67, early_stopping=True))
            models.append(KNeighborsClassifier(n_neighbors=2, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=3, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=4, weights='distance', metric='manhattan'))
            models.append(KNeighborsClassifier(n_neighbors=5, weights='distance', metric='manhattan'))
            models.append(AdaBoostClassifier(learning_rate=0.95, random_state=48, n_estimators=140))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=48, n_estimators=140))
            models.append(AdaBoostClassifier(learning_rate=1.1, random_state=48, n_estimators=140))
            models.append(AdaBoostClassifier(learning_rate=0.95, random_state=16, n_estimators=359))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=16, n_estimators=359))
            models.append(AdaBoostClassifier(learning_rate=1.1, random_state=16, n_estimators=359))
            models.append(AdaBoostClassifier(learning_rate=0.95, random_state=9, n_estimators=257))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=9, n_estimators=257))
            models.append(AdaBoostClassifier(learning_rate=1.1, random_state=9, n_estimators=257))
            models.append(AdaBoostClassifier(learning_rate=0.95, random_state=88, n_estimators=50))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=88, n_estimators=50))
            models.append(AdaBoostClassifier(learning_rate=1.1, random_state=88, n_estimators=50))
            models.append(AdaBoostClassifier(learning_rate=0.95, random_state=48, n_estimators=447))
            models.append(AdaBoostClassifier(learning_rate=1.0, random_state=48, n_estimators=447))
            models.append(AdaBoostClassifier(learning_rate=1.1, random_state=48, n_estimators=447))
        ###############################################



        # models.append(CatBoostClassifier(random_state=42, logging_level='Silent',))
        # models.append(LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000))
        # models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.01, max_iter=1000, random_state=54, early_stopping=True))
        # models.append(MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation='tanh', solver='lbfgs', alpha=0.01, max_iter=1000, random_state=88, early_stopping=True))


        # for k in range(5):
        #     n = k + 2
        #     models.append(KNeighborsClassifier(n_neighbors=n, weights='distance', metric='manhattan'))



        # for k in range(3):
        #     RS = random.randint(1, 100)
        #     n_estimators = random.randint(50,500)

        #     models.append(RandomForestClassifier(random_state=RS, n_estimators=n_estimators))
        #     models.append(MLPClassifier(hidden_layer_sizes=(120, 60, 30), activation='relu', solver='lbfgs', max_iter=1000, random_state=RS))
        #     models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='tanh', solver='lbfgs', alpha=0.001, max_iter=1000, random_state=RS, early_stopping=True))
        #     models.append(MLPClassifier(hidden_layer_sizes=(120, 60), activation='relu', solver='lbfgs', max_iter=1000, random_state=RS))
        #     models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=0.95))
        #     models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=1.1))
        #     models.append(AdaBoostClassifier(random_state=RS, n_estimators=n_estimators, learning_rate=1.0))
        #     models.append(RandomForestClassifier(random_state=RS, n_estimators=n_estimators))






    else: # lista modeli wczytana z pliku
        models = loaded_models
    

    votes_for_first_team = {}
    votes_for_second_team = {}
    votes_for_third_team = {}

    for k, model in enumerate(models):
        if len(loaded_models) == 0:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # show_heatmap(y_test, y_pred)

        vote = 1

        # if type(model).__name__ == 'MLPClassifier' and who == 'ROOKIE':
        #     vote = 5
        # else:
        #     vote = 1

        for pred, player_name in zip(y_pred, player_names):
            if pred == 1:
                votes_for_first_team[player_name] = votes_for_first_team.get(player_name,0) + vote
            elif pred == 2:
                votes_for_second_team[player_name] = votes_for_second_team.get(player_name,0) + vote
            elif pred == 3 and who == 'ALLNBA':
                votes_for_third_team[player_name] = votes_for_third_team.get(player_name,0) + vote


    # sortowanie głosów
    votes_for_first_team = sorted(votes_for_first_team.items(), key=lambda x: x[1], reverse=True)
    votes_for_second_team = sorted(votes_for_second_team.items(), key=lambda x: x[1], reverse=True)
    votes_for_third_team = sorted(votes_for_third_team.items(), key=lambda x: x[1], reverse=True)



    # decydowanie / odrzucanie duplikatów
    first_team = []
    second_team = []
    third_team = []
    players_in_team = set()

    print('\n\n', '=-'*20)
    print(votes_for_first_team)
    print('=-'*20)
    print(votes_for_second_team)
    print('=-'*20)
    print(votes_for_third_team)
    print('=-'*20, '\n\n')

    if who == 'ALLNBA':
        first_team  = fill_team_with_position_limit(votes_for_first_team,  df_data, players_in_team)
        players_in_team.update(first_team)
        second_team = fill_team_with_position_limit(votes_for_second_team, df_data, players_in_team)
        players_in_team.update(second_team)
        third_team  = fill_team_with_position_limit(votes_for_third_team,  df_data, players_in_team)
        players_in_team.update(third_team)
    else:
        for name, votes in votes_for_first_team:
            if name not in players_in_team and len(first_team) < 5:
                first_team.append(name)
                players_in_team.add(name)

        for name, votes in votes_for_second_team:
            if name not in players_in_team and len(second_team) < 5:
                second_team.append(name)
                players_in_team.add(name)



    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f'Czas wykonania {who}: {minutes:.2f} min {seconds:.2f} s')

    print("First team:", first_team)
    print("Second team:", second_team)
    if who == 'ALLNBA':
        print("Third team:", third_team)

    ################## WERYFIKACJA ##################

    print('=='*20)
    print('Real votes for First team:')
    real_votes_for_first_team = df_test[df_test[target] == 1][t_pname].values
    real_votes_for_second_team = df_test[df_test[target] == 2][t_pname].values
    if who == 'ALLNBA':
        real_votes_for_third_team = df_test[df_test[target] == 3][t_pname].values

    check_first_team = set(first_team) & set(real_votes_for_first_team)
    check_second_team = set(second_team) & set(real_votes_for_second_team)
    if who == 'ALLNBA':
        check_third_team = set(third_team) & set(real_votes_for_third_team)
    
    print(f'Pierwsza piatka {who} zgadniętych = {len(check_first_team)}')
    print(f'Druga piatka {who} zgadniętych = {len(check_second_team)}')
    if who == 'ALLNBA':
        print(f'Trzecia piatka {who} zgadniętych = {len(check_third_team)}')

    if who == 'ALLNBA':
        model_result = (len(check_first_team) + len(check_second_team) + len(check_third_team)) / 15
        return model_result, models, first_team, second_team, third_team
    else:
        model_result = (len(check_first_team) + len(check_second_team)) / 10
        return model_result, models, first_team, second_team


    




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_json', type=str, help="Ścieżka absolutna do pliku wynikowego .json")
    args = parser.parse_args()
    output_json_path = Path(args.output_json)


    df_rookie = pd.read_csv('nba_rookie_allstats.csv')
    df_allnba = pd.read_csv('nba_allnba_allstats.csv')


    if USE_SAVED_MODELS:
        with open('models/models_allnba_r0.60%_2025-05-26.pkl', 'rb') as f:
            loaded_models_allnba = pickle.load(f)

        with open('models/models_rookie_r0.80%_2025-05-26.pkl', 'rb') as f:
            loaded_models_rookie = pickle.load(f)

        print('=='*20)
        print_models_parameters(loaded_models_allnba)
        print('=='*20)
        print_models_parameters(loaded_models_rookie)
        print('=='*20)
    else:
        loaded_models_allnba = []
        loaded_models_rookie = []


    


    print('ALLNBA')
    result_allnba, models_allnba, first_team_a, second_team_a, third_team_a = perpare_and_train(df_=df_allnba, who='ALLNBA', test_season=2025, loaded_models=loaded_models_allnba)
    print('\n\nROOKIE')
    result_rookie, models_rookie, first_team_r, second_team_r = perpare_and_train(df_=df_rookie, who='ROOKIE', test_season=2025, loaded_models=loaded_models_rookie)

    print(f'Wynik modeli: {(result_allnba + result_rookie) / 2:.2f}%')

    ################## CZY ZAPISAĆ MODEL ##################
    if not USE_SAVED_MODELS:
        save_model = input(f'Czy zapisać modele? (y/n): ')
        if save_model == 'y':
            with open(f'models/models_allnba_r{result_allnba:.2f}%_{time.strftime("%Y-%m-%d")}.pkl', 'wb') as f:
                pickle.dump(models_allnba, f)
            with open(f'models/models_rookie_r{result_rookie:.2f}%_{time.strftime("%Y-%m-%d")}.pkl', 'wb') as f:
                pickle.dump(models_rookie, f)


    results_dict = {
        "first all-nba team": first_team_a,
        "second all-nba team": second_team_a,
        "third all-nba team": third_team_a,
        "first rookie all-nba team": first_team_r,
        "second rookie all-nba team": second_team_r
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)
    print(f"Zapisano wyniki predyckji do pliku: {output_json_path}")




if __name__ == "__main__":
    main()
