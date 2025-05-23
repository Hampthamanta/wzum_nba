from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueStandings, commonplayerinfo
from nba_api.stats.static import players
import pandas as pd
import glob
import re
import time


def download_season():
    for y in range(9, 25):
        if y == 9:
            season = f'200{y-1}-{y}'
        else:
            season = f'20{y-1}-{y}'
        basic = LeagueDashPlayerStats(season=season, season_type_all_star='Regular Season')
        basic_df = basic.get_data_frames()[0]
        basic_df.to_csv(f"nba_basic_stats_{season}.csv", index=False)
        print(basic_df.columns)


def download_results():
    # ALLNBA
    df = pd.read_html('https://www.basketball-reference.com/awards/all_league.html')[0]

    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(2011, 2025)]  # 2010-11, ..., 2023-24
    df = df[df['Season'].isin(seasons)]

    df.to_csv('allnba_teams_2011-2024.csv', index=False)
    print("Zapisano do allnba_teams_2011-2024.csv")

    # ROOKIE
    df = pd.read_html('https://www.basketball-reference.com/awards/all_rookie.html')[0]

    # Filtrowanie tylko sezonów 2011-2024
    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(2011, 2025)]
    df = df[df['Season'].isin(seasons)]

    df.to_csv('rookie_teams_2011-2024.csv', index=False)
    print("Zapisano do rookie_teams_2011-2024.csv")


def merge_seasons():
    files = sorted(glob.glob("nba_basic_stats_*.csv"))

    all_dfs = []

    for fname in files:
        # Wyciągnij sezon z nazwy pliku, np. 2010-11 albo 2024-25
        m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
        if m:
            season_end = int(m.group(1)[:2] + m.group(2)) if int(m.group(2)) < 50 else int(m.group(1)[:2] + m.group(2))
            # Np. dla 2010-11 to sezon_end = 2011, dla 2023-24 to sezon_end = 2024
            season = int(m.group(1)) + 1  # Tak robi Basketball Reference
        else:
            # Jeśli nie pasuje, np. 2024-25
            m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
            if m:
                season = int(m.group(1)) + 1
            else:
                # Spróbuj jeszcze innego formatu, np. 2024-25
                m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
                season = int(m.group(1)) + 1 if m else None

        if season is None:
            print(f"Nie można wyciągnąć sezonu z pliku: {fname}")
            continue

        df = pd.read_csv(fname)
        df['SEASON'] = season
        all_dfs.append(df)

    # Połącz wszystkie
    all_basic = pd.concat(all_dfs, ignore_index=True)
    all_basic.to_csv("nba_basic_stats_all_seasons.csv", index=False)

    print(f"Połączono {len(files)} plików. Zapisano nba_basic_stats_all_seasons.csv.")



def fill_results():
    # Wczytaj główny plik ze statystykami
    df = pd.read_csv('nba_basic_stats_all_seasons.csv')

    # Zainicjalizuj nową kolumnę zerami
    df['result_top_five'] = 0

    # Wczytaj listy ALL-NBA i ALL-ROOKIE
    allnba = pd.read_csv('allnba_teams_2011-2024.csv')
    rookie = pd.read_csv('rookie_teams_2011-2024.csv')

    # Funkcja do mapowania sezonu '2023-24' -> 2024 (int)
    def season_str_to_year(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    # Helper do wycinania pozycji (np. 'Joel Embiid C' -> 'Joel Embiid')
    def strip_position(player_with_pos):
        parts = player_with_pos.strip().split()
        # Jeśli ostatni element to pojedyncza litera (C, F, G), to go usuń
        if parts and parts[-1] in ['C', 'F', 'G']:
            return ' '.join(parts[:-1])
        return player_with_pos.strip()

    def update_result_top_five(df, team_df, column='result_top_five'):
        for idx, row in team_df.iterrows():
            try:
                season = season_str_to_year(row['Season'])
            except Exception:
                continue  # pomiń wiersz jeśli błąd
            team = str(row['Tm'])
            for i in range(4, 9):
                player_with_pos = row.get(f'Unnamed: {i}', None)
                if pd.isna(player_with_pos) or not isinstance(player_with_pos, str) or not player_with_pos.strip():
                    continue
                player_name = strip_position(player_with_pos).lower()
                # Używamy dokładnego porównania imię nazwisko (lower)
                mask = (df['SEASON'] == season) & (df['PLAYER_NAME'].str.lower() == player_name)
                # Wpisz odpowiednią wartość do kolumny
                if team == '1st':
                    df.loc[mask, column] = 1
                elif team == '2nd':
                    df.loc[mask, column] = 2
                elif team == '3rd':
                    df.loc[mask, column] = 3

    # Uzupełnij All-NBA Teams (tylko lata 2011-2024)
    update_result_top_five(df, allnba)

    # Uzupełnij All-Rookie Teams (również 1st, 2nd)
    update_result_top_five(df, rookie)

    # Zapisz efekt
    df.to_csv('nba_basic_stats_all_seasons_with_results.csv', index=False)
    print('Zapisano do nba_basic_stats_all_seasons_with_results.csv')


def separate_rooke():
    df = pd.read_csv("nba_basic_stats_all_seasons_with_results.csv")
    df['SEASON'] = df['SEASON'].astype(int)

    # 1. Oznacz graczy, którzy zagrali w sezonie 2010 (czyli 2009-10)
    old_players = set(df[df['SEASON'] == 2010]['PLAYER_ID'])

    # 2. Zostaw tylko sezony 2011 i wyższe
    df = df[df['SEASON'] > 2010]

    # 3. Rookie: pierwszy sezon gracza po 2010, który NIE grał w 2010
    rookie_mask = (~df['PLAYER_ID'].isin(old_players)) & (
        df.groupby('PLAYER_ID')['SEASON'].transform('min') == df['SEASON']
    )
    rookies = df[rookie_mask]

    # 4. allnba: reszta, czyli wszyscy inni (albo grali w 2010, albo to nie jest ich pierwszy sezon)
    allnba = df[~rookie_mask]

    # 5. Zapisz
    rookies.to_csv("nba_basic_stats_rookie.csv", index=False)
    allnba.to_csv("nba_basic_stats_allnba.csv", index=False)

    print(f"Rookie: {len(rookies)} wierszy, allnba: {len(allnba)} wierszy")



def download_teams_stats():
    def get_team_standings(season):
        try:
            standings = LeagueStandings(season=season)
            df = standings.get_data_frames()[0]
            df['SEASON'] = season
            return df
        except Exception as e:
            print(f"Błąd dla sezonu {season}: {e}")
            return pd.DataFrame()

    all_standings = []
    seasons = [
        f"{year}-{str(year+1)[-2:]}" for year in range(2010, 2025)
    ]

    for season in seasons:
        print(f"Pobieram dane dla sezonu {season}...")
        df = get_team_standings(season)
        if not df.empty:
            all_standings.append(df)
        time.sleep(1)  # żeby nie przeciążyć API

    if all_standings:
        all_seasons_df = pd.concat(all_standings, ignore_index=True)
        all_seasons_df.to_csv('nba_team_standings_2011_2025.csv', index=False)
        print("Zapisano do nba_team_standings_2011_2025.csv")
    else:
        print("Nie pobrano żadnych danych.")



def fill_teams_data_allnba():
    # Wczytaj pliki
    df_teams = pd.read_csv('nba_team_standings_2011_2025.csv')
    df_players = pd.read_csv('nba_basic_stats_allnba.csv')

    # Konwersja sezonów: '2010-11' -> 2011, '2011-12' -> 2012 itd.
    def season_str_to_int(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    df_teams['SEASON'] = df_teams['SEASON'].apply(season_str_to_int)

    # Ujednolicamy nazwy kolumn kluczowych
    df_teams = df_teams.rename(columns={'TeamID': 'TEAM_ID'})
    # Upewniamy się, że typy są zgodne (int)
    df_teams['TEAM_ID'] = df_teams['TEAM_ID'].astype(int)
    df_players['TEAM_ID'] = df_players['TEAM_ID'].astype(int)

    # Przygotuj do joinowania po sezonie i TEAM_ID
    cols_to_add = ['TEAM_ID', 'SEASON', 'PlayoffRank', 'WINS', 'LongWinStreak']
    df_teams_subset = df_teams[cols_to_add]

    # Łączymy dane gracza z danymi drużynowymi po TEAM_ID i SEASON
    df_merged = df_players.merge(
        df_teams_subset,
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )

    # Zapisz wynik
    df_merged.to_csv('nba_basic_stats_allnba.csv', index=False)
    print("Zapisano do nba_basic_stats_allnba.csv")


def fill_teams_data_rookie():
    # Wczytaj pliki
    df_teams = pd.read_csv('nba_team_standings_2011_2025.csv')
    df_players = pd.read_csv('nba_basic_stats_rookie.csv')

    # Konwersja sezonów: '2010-11' -> 2011, '2011-12' -> 2012 itd.
    def season_str_to_int(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    df_teams['SEASON'] = df_teams['SEASON'].apply(season_str_to_int)

    # Ujednolicamy nazwy kolumn kluczowych
    df_teams = df_teams.rename(columns={'TeamID': 'TEAM_ID'})
    # Upewniamy się, że typy są zgodne (int)
    df_teams['TEAM_ID'] = df_teams['TEAM_ID'].astype(int)
    df_players['TEAM_ID'] = df_players['TEAM_ID'].astype(int)

    # Przygotuj do joinowania po sezonie i TEAM_ID
    cols_to_add = ['TEAM_ID', 'SEASON', 'PlayoffRank', 'WINS', 'LongWinStreak']
    df_teams_subset = df_teams[cols_to_add]

    # Łączymy dane gracza z danymi drużynowymi po TEAM_ID i SEASON
    df_merged = df_players.merge(
        df_teams_subset,
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )

    # Zapisz wynik
    df_merged.to_csv('nba_basic_stats_rookie.csv', index=False)
    print("Zapisano do nba_basic_stats_rookie.csv")


def add_positions():
    def get_position_for_player(player_id):
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_normalized_dict()
        pos = info['CommonPlayerInfo'][0]['POSITION']
        return pos

    # Dla wszystkich unikalnych PLAYER_ID:
    df = pd.read_csv('nba_basic_stats_allnba.csv')
    unique_ids = df['PLAYER_ID'].unique()
    positions = {}
    for pid in unique_ids:
        try:
            positions[pid] = get_position_for_player(pid)
        except Exception as e:
            positions[pid] = None

    # Przypisz do DataFrame
    df['POSITION'] = df['PLAYER_ID'].map(positions)
    df.to_csv('nba_basic_stats_allnba_with_position.csv', index=False)



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
        print(corr_sorted)
    
    return corr_sorted
    


if __name__ == "__main__":
    print('Przygotowanie danych NBA')

    add_positions()


    if 0:
        download_season()
        download_results()
        merge_seasons()
        fill_results()
        separate_rooke()
        download_teams_stats()
        fill_teams_data_allnba()
        fill_teams_data_rookie()