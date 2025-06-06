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


def calculate_correlations(df ,top=20):
    target='result_top_five'

    # Wybierz tylko kolumny numeryczne
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Usuń z cech identyfikatory i wynik
    exclude = ['PLAYER_ID', 'SEASON', target]
    features = [col for col in numeric_cols if col not in exclude]
    
    # Wylicz korelacje
    corr = df[features + [target]].corr()[target].drop(target)
    
    # Posortuj malejąco po wartości bezwzględnej
    corr_sorted = corr.abs().sort_values(ascending=False)
    
    # Wyświetl top cech (lub wszystkie jeśli top=None)
    if top is not None:
        print(corr_sorted.head(top))
    else:
        for feature, value in corr_sorted.items():
            print(f'{feature}: {value:.4f}')



def calculate_correlations_plot(df, top=20):
    target = 'result_top_five'

    # Wybierz tylko kolumny numeryczne
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Usuń z cech identyfikatory i wynik
    exclude = ['PLAYER_ID', 'SEASON', target]
    features = [col for col in numeric_cols if col not in exclude]
    
    # Wylicz korelacje
    corr = df[features + [target]].corr()[target].drop(target)
    
    # Posortuj malejąco po wartości bezwzględnej
    corr_sorted = corr.abs().sort_values(ascending=False)

    # Wyświetl wykres dla top cech
    if top is not None:
        corr_sorted = corr_sorted.head(top)
    
    plt.figure(figsize=(10, max(4, 0.4*len(corr_sorted))))
    corr_sorted[::-1].plot(kind='barh')  # Odwróć, by najwyższa wartość była na górze
    plt.title(f'Top {len(corr_sorted)} cech najbardziej skorelowanych z {target}')
    plt.xlabel('Wartość bezwzględna korelacji')
    plt.tight_layout()
    plt.show()
    


def print_models_parameters(models):
    for model in models:
        print(type(model).__name__)
        if type(model).__name__ == 'KNeighborsClassifier':
            print(f'\t n_neighbors = {model.n_neighbors}')
            print(f'\t weights = {model.weights}')
            print(f'\t metric = {model.metric}')

        elif type(model).__name__ == 'MLPClassifier':
            print(f'\t hidden_layer_sizes = {model.hidden_layer_sizes}')
            print(f'\t activation = {model.activation}')
            print(f'\t solver = {model.solver}')
            print(f'\t alpha = {model.alpha}')
            print(f'\t max_iter = {model.max_iter}')
            print(f'\t random_state = {model.random_state}')
            print(f'\t early_stopping = {model.early_stopping}')

        elif type(model).__name__ == 'LogisticRegression':
            print(f'\t n_neighbors = {model.max_iter}')
            print(f'\t weights = {model.solver}')
            print(f'\t metric = {model.random_state}')

        elif type(model).__name__ == 'RandomForestClassifier':
            print(f'\t n_estimators = {model.n_estimators}')
            print(f'\t metric = {model.random_state}')

        elif type(model).__name__ == 'AdaBoostClassifier':
            print(f'\t learning_rate = {model.learning_rate}')
            print(f'\t random_state = {model.random_state}')
            print(f'\t n_estimators = {model.n_estimators}')

