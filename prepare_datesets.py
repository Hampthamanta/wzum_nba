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

    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(2011, 2026)]  # 2010-11, ..., 2023-24
    df = df[df['Season'].isin(seasons)]

    df.to_csv('allnba_teams_2011-2024.csv', index=False)
    print("Zapisano do allnba_teams_2011-2024.csv")

    # ROOKIE
    df = pd.read_html('https://www.basketball-reference.com/awards/all_rookie.html')[0]

    seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(2011, 2026)]
    df = df[df['Season'].isin(seasons)]

    df.to_csv('rookie_teams_2011-2024.csv', index=False)
    print("Zapisano do rookie_teams_2011-2024.csv")


def merge_seasons():
    files = sorted(glob.glob("nba_basic_stats_*.csv"))

    all_dfs = []

    for fname in files:
        m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
        if m:
            season = int(m.group(1)) + 1
        else:
            m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
            if m:
                season = int(m.group(1)) + 1
            else:
                m = re.search(r'nba_basic_stats_(\d{4})-(\d{2})\.csv', fname)
                season = int(m.group(1)) + 1 if m else None

        if season is None:
            print(f"Nie można wyciągnąć sezonu z pliku: {fname}")
            continue

        df = pd.read_csv(fname)
        df['SEASON'] = season
        all_dfs.append(df)

    all_basic = pd.concat(all_dfs, ignore_index=True)
    all_basic.to_csv("nba_basic_stats_all_seasons.csv", index=False)

    print(f"Połączono {len(files)} plików. Zapisano nba_basic_stats_all_seasons.csv.")



def fill_results():
    df = pd.read_csv('nba_basic_stats_all_seasons.csv')
    df['result_top_five'] = 0

    allnba = pd.read_csv('allnba_teams_2011-2024.csv')
    rookie = pd.read_csv('rookie_teams_2011-2024.csv')

    def season_str_to_year(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    def strip_position(player_with_pos):
        parts = player_with_pos.strip().split()
        if parts and parts[-1] in ['C', 'F', 'G']:
            return ' '.join(parts[:-1])
        return player_with_pos.strip()

    def update_result_top_five(df, team_df, column='result_top_five'):
        for idx, row in team_df.iterrows():
            try:
                season = season_str_to_year(row['Season'])
            except Exception:
                continue
            team = str(row['Tm'])
            for i in range(4, 9):
                player_with_pos = row.get(f'Unnamed: {i}', None)
                if pd.isna(player_with_pos) or not isinstance(player_with_pos, str) or not player_with_pos.strip():
                    continue
                player_name = strip_position(player_with_pos).lower()
                mask = (df['SEASON'] == season) & (df['PLAYER_NAME'].str.lower() == player_name)

                if team == '1st':
                    df.loc[mask, column] = 1
                elif team == '2nd':
                    df.loc[mask, column] = 2
                elif team == '3rd':
                    df.loc[mask, column] = 3


    update_result_top_five(df, allnba)
    update_result_top_five(df, rookie)

    df.to_csv('nba_basic_stats_all_seasons_with_results.csv', index=False)
    print('Zapisano do nba_basic_stats_all_seasons_with_results.csv')


def separate_rooke():
    df = pd.read_csv("nba_basic_stats_all_seasons_with_results.csv")
    df['SEASON'] = df['SEASON'].astype(int)

    old_players = set(df[df['SEASON'] == 2010]['PLAYER_ID'])
    df = df[df['SEASON'] > 2010]

    rookie_mask = (~df['PLAYER_ID'].isin(old_players)) & (
        df.groupby('PLAYER_ID')['SEASON'].transform('min') == df['SEASON']
    )
    rookies = df[rookie_mask]
    allnba = df[~rookie_mask]

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
        time.sleep(1)

    if all_standings:
        all_seasons_df = pd.concat(all_standings, ignore_index=True)
        all_seasons_df.to_csv('nba_team_standings_2011_2025.csv', index=False)
        print("Zapisano do nba_team_standings_2011_2025.csv")
    else:
        print("Nie pobrano żadnych danych.")



def fill_teams_data_allnba():
    df_teams = pd.read_csv('nba_team_standings_2011_2025.csv')
    df_players = pd.read_csv('nba_basic_stats_allnba.csv')

    def season_str_to_int(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    df_teams['SEASON'] = df_teams['SEASON'].apply(season_str_to_int)
    df_teams = df_teams.rename(columns={'TeamID': 'TEAM_ID'})
    df_teams['TEAM_ID'] = df_teams['TEAM_ID'].astype(int)
    df_players['TEAM_ID'] = df_players['TEAM_ID'].astype(int)

    cols_to_add = ['TEAM_ID', 'SEASON', 'PlayoffRank', 'WINS', 'LongWinStreak']
    df_teams_subset = df_teams[cols_to_add]

    df_merged = df_players.merge(
        df_teams_subset,
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )

    df_merged.to_csv('nba_basic_stats_allnba.csv', index=False)
    print("Zapisano do nba_basic_stats_allnba.csv")


def fill_teams_data_rookie():
    df_teams = pd.read_csv('nba_team_standings_2011_2025.csv')
    df_players = pd.read_csv('nba_basic_stats_rookie.csv')

    def season_str_to_int(season_str):
        base, end = season_str.split('-')
        return int(base) + 1

    df_teams['SEASON'] = df_teams['SEASON'].apply(season_str_to_int)
    df_teams = df_teams.rename(columns={'TeamID': 'TEAM_ID'})
    df_teams['TEAM_ID'] = df_teams['TEAM_ID'].astype(int)
    df_players['TEAM_ID'] = df_players['TEAM_ID'].astype(int)

    cols_to_add = ['TEAM_ID', 'SEASON', 'PlayoffRank', 'WINS', 'LongWinStreak']
    df_teams_subset = df_teams[cols_to_add]

    df_merged = df_players.merge(
        df_teams_subset,
        on=['TEAM_ID', 'SEASON'],
        how='left'
    )

    df_merged.to_csv('nba_basic_stats_rookie.csv', index=False)
    print("Zapisano do nba_basic_stats_rookie.csv")


def add_positions():
    def get_position_for_player(player_id):
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_normalized_dict()
        pos = info['CommonPlayerInfo'][0]['POSITION']
        return pos

    df = pd.read_csv('nba_basic_stats_allnba.csv')
    unique_ids = df['PLAYER_ID'].unique()
    positions = {}
    for pid in unique_ids:
        try:
            positions[pid] = get_position_for_player(pid)
        except Exception as e:
            positions[pid] = None

    df['POSITION'] = df['PLAYER_ID'].map(positions)
    df.to_csv('nba_basic_stats_allnba_with_position.csv', index=False)


    

def download_advanced_stats(start_season=2011, end_season=2025, sleep_sec=2):
    all_dfs = []
    for year in range(start_season, end_season+1):
        url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
        print(f"Pobieram: {url}")
        try:
            df = pd.read_html(url, header=0)[0]
            df = df[df['Player'] != 'Player']
            df['Season'] = f"{year-1}-{str(year)[-2:]}"
            all_dfs.append(df)
            df.to_csv(f"nba_advanced_{year}.csv", index=False)
            print(f"Zapisano: nba_advanced_{year}.csv")
        except Exception as e:
            print(f"Błąd pobierania {url}: {e}")
        time.sleep(sleep_sec)

    if all_dfs:
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv("nba_advanced_2011-2025.csv", index=False)
        print("Zapisano pełny zbiór: nba_advanced_2011-2025.csv")
    return



def add_advanced_stats(who='allnba'):
    import pandas as pd

    basic = pd.read_csv(f"nba_basic_stats_{who}.csv")
    adv = pd.read_csv("nba_advanced_2011-2025.csv")

    def season_int_to_str(season):
        season = int(season)
        return f"{season-1}-{str(season)[-2:]}"

    basic["Season"] = basic["SEASON"].apply(season_int_to_str)


    def simplify_name(x):
        return (
            str(x)
            .replace('.', '')
            .replace('-', '')
            .replace("’", "'")
            .replace('`', "'")
            .replace('’', "'")
            .replace("’", "'")
            .replace(",", "")
            .replace(" Jr", "")
            .replace(" II", "")
            .replace(" III", "")
            .replace(" IV", "")
            .replace(" V", "")
            .lower()
            .strip()
        )

    basic['match_name'] = basic['PLAYER_NAME'].apply(simplify_name)
    adv['match_name'] = adv['Player'].apply(simplify_name)
    adv = adv.drop_duplicates(subset=['match_name', 'Season'], keep='first')

    joined = pd.merge(
        basic,
        adv,
        left_on=['match_name', 'Season'],
        right_on=['match_name', 'Season'],
        suffixes=('_basic', '_adv'),
        how='inner'
    )

    adv_features = [
        'Pos', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%', 'DRB%', 'TRB%', 'AST%',
        'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/48',
        'OBPM', 'DBPM', 'BPM', 'VORP'
    ]
    final_cols = list(basic.columns) + [col for col in adv_features if col in adv.columns]

    final = joined[final_cols]
    final = final.drop(columns=['Season', 'match_name'], errors='ignore')

    final.to_csv(f"nba_{who}_allstats.csv", index=False)
    print(f"Zapisano do nba_{who}_allstats.csv")





if __name__ == "__main__":
    print('Przygotowanie danych NBA')

    if 0:
        download_season()
        download_results()
        merge_seasons()
        fill_results()
        separate_rooke()
        download_teams_stats()
        fill_teams_data_allnba()
        fill_teams_data_rookie()


    # download_advanced_stats(start_season=2011, end_season=2025, sleep_sec=2)
    add_advanced_stats(who='allnba')