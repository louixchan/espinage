import pandas as pd
import numpy as np
import datetime
import os

import pickle

import sklearn.utils
from scipy.stats.mstats import gmean

TARGET = 'P1'

NORMALISED_DATA_PATH = f'../data/{TARGET}/normalised'

FIXTURE_DATA_PATH = f'../data/fixtures'

SETTINGS_PATH = f'../data/{TARGET}/settings'

DATE_FORMAT = '%Y-%m-%d'

ODDS = [
	'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'LBH', 'LBD', 'LBA',
	'VCH', 'VCD', 'VCA', 'BbMxH', 'BbAvH', 'BbMxD', 'BbAvD', 'BbMxA',
	'BbAvA', 'BbMx>2.5', 'BbAv>2.5', 'BbMx<2.5', 'BbAv<2.5',
	'BbMxAHH', 'BbAvAHH', 'BbMxAHA', 'BbAvAHA', 'IWH', 'IWD', 'IWA'
]

BOOKIES = [
	'B365', 'BW', 'IW', 'LB', 'VC', 'BbMx', 'BbAv'
]

COLUMNS = []

with open(os.path.join(SETTINGS_PATH, 'raw-columns')) as f:

	COLUMNS = f.read().split('\n')

	if COLUMNS[-1] == '':

		COLUMNS = COLUMNS[:-1]

MATCH_STAT_COLUMNS = []

MATCH_RESULT_STAT_COLUMNS = []

TEAM_STAT_COLUMNS = []

ODDS_COLUMNS = []



def unit_vector(vector):
	""" Returns the unit vector of the vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):

	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def get_match_data(data):

	df_1 = data[['Date', 'Season_of_league', 'HomeTeam'] + MATCH_RESULT_STAT_COLUMNS].rename(
		columns={'HomeTeam': 'team'})
	df_2 = data[['Date', 'Season_of_league', 'AwayTeam'] + MATCH_RESULT_STAT_COLUMNS].rename(
		columns={'AwayTeam': 'team'})
	df_2['HTR-0'] = -df_2['HTR-0']
	df_2['FTR-0'] = -df_2['FTR-0']

	if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:
		df_2['FT-GD'] = -df_2['FT-GD']

	if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:
		df_2['HT-GD'] = -df_2['HT-GD']

	df_1 = df_1.append(df_2).sort_values(by='Date')

	if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:

		df_1['FT-GF'] = np.where(df_1['FT-GD'] > 0, df_1['FT-GD'], 0)
		df_1['FT-GA'] = np.where(df_1['FT-GD'] < 0, -df_1['FT-GD'], 0)

	if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:

		df_1['HT-GF'] = np.where(df_1['HT-GD'] > 0, df_1['HT-GD'], 0)
		df_1['HT-GA'] = np.where(df_1['HT-GD'] < 0, -df_1['HT-GD'], 0)

	if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:

		df_1['FTR-0-win'] = np.where(df_1['FTR-0'] == 1, 1, 0)
		df_1['FTR-0-draw'] = np.where(df_1['FTR-0'] == 0, 1, 0)
		df_1['FTR-0-lose'] = np.where(df_1['FTR-0'] == -1, 1, 0)

	if 'HTR-0' in MATCH_RESULT_STAT_COLUMNS:

		df_1['HTR-0-win'] = np.where(df_1['HTR-0'] == 1, 1, 0)
		df_1['HTR-0-draw'] = np.where(df_1['HTR-0'] == 0, 1, 0)
		df_1['HTR-0-lose'] = np.where(df_1['HTR-0'] == -1, 1, 0)

	df_1.drop_duplicates(inplace=True)

	return df_1


def get_season_data(data):

	global MATCH_RESULT_STAT_COLUMNS, MATCH_STAT_COLUMNS, TEAM_STAT_COLUMNS

	df_match = None

	if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:

		TEAM_STAT_COLUMNS.extend([
			'point-expanding', 'point-ewm-0.15', 'point-ewm-0.25', 'point-ewm-0.75',
			'point-ewm-0.5', 'point-ewm-0.9', 'point-rolling-3', 'FTR-0-win-rate',
			'FTR-0-draw-rate', 'FTR-0-lose-rate', 'FTR-0-win-rate_5', 'FTR-0-draw-rate_5',
			'FTR-0-lose-rate_5', 'point_last'
		])

	if 'HTR-0' in MATCH_RESULT_STAT_COLUMNS:

		TEAM_STAT_COLUMNS.extend([
			'HTR-0-win-rate', 'HTR-0-draw-rate', 'HTR-0-lose-rate', 'HTR-0-win-rate_5', 'HTR-0-draw-rate_5',
			'HTR-0-lose-rate_5', 'HTR-0-draw-rate_last', 'HTR-0-lose-rate_last', 'HTR-0-win-rate_last'
		])

	if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:

		TEAM_STAT_COLUMNS += [
			'HT-GD-sum', 'HT-GF-sum', 'HT-GA-sum', 'HT-GF-GA-expanding', 'HT-GD_last', 'HT-GF_last', 'HT-GA_last'
		]

	if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:

		TEAM_STAT_COLUMNS += [
			'FT-GD-sum', 'FT-GF-sum', 'FT-GA-sum', 'FT-GF-GA-expanding', 'FT-GD_last', 'HT-GF_last', 'FT-GA_last'
		]

	for season in data.Season_of_league.unique():

		print(season)

		df_temp = data.query('Season_of_league == ' + str(season))
		df_temp.drop_duplicates(inplace=True)

		for team in df_temp.team.unique():

			df_temp_team = df_temp.query('team == "' + team + '"').sort_values(by='Date').reset_index(
				drop=True).reset_index()

			if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
				df_temp_team['point'] = np.where(
					df_temp_team['FTR-0'] == 1,
					3,
					np.where(
						df_temp_team['FTR-0'] == 0,
						1,
						0
					)
				)
				df_temp_team['point'] = df_temp_team['point'].shift(1)
				df_temp_team['point-rolling-3'] = df_temp_team['point'].rolling(3).sum()
				df_temp_team['point-expanding'] = df_temp_team['point'].expanding(1).sum()
				df_temp_team['point-expanding'] = df_temp_team['point'].fillna(0)
				df_temp_team['point-ewm-0.9'] = df_temp_team['point-expanding'].ewm(alpha=0.9, adjust=False).mean()
				df_temp_team['point-ewm-0.75'] = df_temp_team['point-expanding'].ewm(alpha=0.75, adjust=False).mean()
				df_temp_team['point-ewm-0.5'] = df_temp_team['point-expanding'].ewm(alpha=0.5, adjust=False).mean()
				df_temp_team['point-ewm-0.25'] = df_temp_team['point-expanding'].ewm(alpha=0.25, adjust=False).mean()
				df_temp_team['point-ewm-0.15'] = df_temp_team['point-expanding'].ewm(alpha=0.15, adjust=False).mean()

			if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:
				df_temp_team['HT-GD-sum'] = df_temp_team['HT-GD'].shift(1).expanding(1).sum()
				df_temp_team['HT-GF-sum'] = df_temp_team['HT-GF'].shift(1).expanding(1).sum()
				df_temp_team['HT-GA-sum'] = df_temp_team['HT-GF'].shift(1).expanding(1).sum()
				df_temp_team['HT-GF-GA-expanding'] = df_temp_team['HT-GF-sum'] / df_temp_team['HT-GA-sum']

			if 'HTR-0' in MATCH_RESULT_STAT_COLUMNS:
				df_temp_team['HTR-0-win-rate'] = df_temp_team['HTR-0-win'].shift(1).expanding(1).sum() / df_temp_team[
					'index']
				df_temp_team['HTR-0-draw-rate'] = df_temp_team['HTR-0-draw'].shift(1).expanding(1).sum() / df_temp_team[
					'index']
				df_temp_team['HTR-0-lose-rate'] = df_temp_team['HTR-0-lose'].shift(1).expanding(1).sum() / df_temp_team[
					'index']

				df_temp_team['HTR-0-win-rate_5'] = df_temp_team['HTR-0-win'].shift(1).rolling(5,
				                                                                              min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)
				df_temp_team['HTR-0-draw-rate_5'] = df_temp_team['HTR-0-draw'].shift(1).rolling(5,
				                                                                                min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)
				df_temp_team['HTR-0-lose-rate_5'] = df_temp_team['HTR-0-lose'].shift(1).rolling(5,
				                                                                                min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)

			if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:
				df_temp_team['FT-GD-sum'] = df_temp_team['FT-GD'].shift(1).expanding(1).sum()
				df_temp_team['FT-GF-sum'] = df_temp_team['FT-GF'].shift(1).expanding(1).sum()
				df_temp_team['FT-GA-sum'] = df_temp_team['FT-GF'].shift(1).expanding(1).sum()
				df_temp_team['FT-GF-GA-expanding'] = df_temp_team['FT-GF-sum'] / df_temp_team['FT-GA-sum']

			if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
				df_temp_team['FTR-0-win-rate'] = df_temp_team['FTR-0-win'].shift(1).expanding(1).sum() / df_temp_team[
					'index']
				df_temp_team['FTR-0-draw-rate'] = df_temp_team['FTR-0-draw'].shift(1).expanding(1).sum() / df_temp_team[
					'index']
				df_temp_team['FTR-0-lose-rate'] = df_temp_team['FTR-0-lose'].shift(1).expanding(1).sum() / df_temp_team[
					'index']

				df_temp_team['FTR-0-win-rate_5'] = df_temp_team['FTR-0-win'].shift(1).rolling(5,
				                                                                              min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)
				df_temp_team['FTR-0-draw-rate_5'] = df_temp_team['FTR-0-draw'].shift(1).rolling(5,
				                                                                                min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)
				df_temp_team['FTR-0-lose-rate_5'] = df_temp_team['FTR-0-lose'].shift(1).rolling(5,
				                                                                                min_periods=1).sum() / np.where(
					df_temp_team['index'] < 5, df_temp_team['index'], 5)

			if len(data[(data.Season_of_league == season - 1) & (data.team == team)]) > 0:

				df_temp_last_season = data[
					(data.Season_of_league == season - 1) & (data.team == team)]

				if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_last_season['point'] = np.where(
						df_temp_last_season['FTR-0'] == 1,
						3,
						np.where(
							df_temp_last_season['FTR-0'] == 0,
							1,
							0
						)
					)
					df_temp_team['point_last'] = df_temp_last_season['point'].sum()

				if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['HT-GD_last'] = df_temp_last_season['HT-GD'].sum()
					df_temp_team['HT-GF_last'] = df_temp_last_season['HT-GF'].sum()
					df_temp_team['HT-GA_last'] = df_temp_last_season['HT-GF'].sum()

				if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['FT-GD_last'] = df_temp_last_season['FT-GD'].sum()
					df_temp_team['FT-GF_last'] = df_temp_last_season['FT-GF'].sum()
					df_temp_team['FT-GA_last'] = df_temp_last_season['FT-GF'].sum()

				if 'HTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['HTR-0-win-rate_last'] = df_temp_last_season['HTR-0-win'].sum() / len(
						df_temp_last_season)
					df_temp_team['HTR-0-draw-rate_last'] = df_temp_last_season['HTR-0-draw'].sum() / len(
						df_temp_last_season)
					df_temp_team['HTR-0-lose-rate_last'] = df_temp_last_season['HTR-0-lose'].sum() / len(
						df_temp_last_season)

				if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['FTR-0-win-rate_last'] = df_temp_last_season['FTR-0-win'].sum() / len(
						df_temp_last_season)
					df_temp_team['FTR-0-draw-rate_last'] = df_temp_last_season['FTR-0-draw'].sum() / len(
						df_temp_last_season)
					df_temp_team['FTR-0-lose-rate_last'] = df_temp_last_season['FTR-0-lose'].sum() / len(
						df_temp_last_season)

			else:

				if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['point_last'] = -1

				if 'HT-GD' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['HT-GD_last'] = -1
					df_temp_team['HT-GF_last'] = -1
					df_temp_team['HT-GA_last'] = -1

				if 'FT-GD' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['FT-GD_last'] = -1
					df_temp_team['FT-GF_last'] = -1
					df_temp_team['FT-GA_last'] = -1

				if 'HTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['HTR-0-win-rate_last'] = -1
					df_temp_team['HTR-0-draw-rate_last'] = -1
					df_temp_team['HTR-0-lose-rate_last'] = -1

				if 'FTR-0' in MATCH_RESULT_STAT_COLUMNS:
					df_temp_team['FTR-0-win-rate_last'] = -1
					df_temp_team['FTR-0-draw-rate_last'] = -1
					df_temp_team['FTR-0-lose-rate_last'] = -1

			if df_match is None:

				df_match = df_temp_team[['index', 'Date', 'team'] + TEAM_STAT_COLUMNS]\
					.rename(columns={'index': 'match_in_league'})

			else:

				df_match = df_match.append(df_temp_team[['index', 'Date', 'team'] + TEAM_STAT_COLUMNS]\
					.rename(columns={'index': 'match_in_league'}))

	return df_match


df = pd.read_csv(os.path.join(NORMALISED_DATA_PATH, 'data-concat.txt'), delimiter='\t').reset_index(drop=True)

df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT)
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['Day_of_week'] = df['Date'].dt.dayofweek
df['Season_of_year'] = (df['Month'] % 12 + 3)//3
df['Season_of_league'] = np.where(
	df['Month'] < 7,
	df['Date'].dt.year - 1,
	df['Date'].dt.year
)

if 'FTHG' in COLUMNS:

	df['FT-GD'] = df['FTHG'] - df['FTAG']
	df['FTG'] = df['FTHG'] + df['FTAG']

	df['FTG-2.5'] = np.where(
		df['FTG'] > 2.5,
		1,
		0
	)

	MATCH_RESULT_STAT_COLUMNS.append('FTG')
	MATCH_RESULT_STAT_COLUMNS.append('FT-GD')
	MATCH_RESULT_STAT_COLUMNS.append('FTG-2.5')

if 'HTHG' in COLUMNS:

	df['HT-GD'] = df['HTHG'] - df['HTAG']
	df['HTG'] = df['HTHG'] + df['HTAG']

	df['HTG-1.5'] = np.where(
		df['HTG'] > 1.5,
		1,
		0
	)

	MATCH_RESULT_STAT_COLUMNS.append('HTG')
	MATCH_RESULT_STAT_COLUMNS.append('HT-GD')
	MATCH_RESULT_STAT_COLUMNS.append('HTG-1.5')

if 'FTR' in COLUMNS:

	df['FTR-0'] = np.where(
		df['FTR'] == 'H',
		1,
		np.where(
			df['FTR'] == 'D',
			0,
			-1
		)
	)
	df['FTR-H'] = np.where(
		df['FTR'] == 'H',
		1,
		0
	)
	df['FTR-D'] = np.where(
		df['FTR'] == 'D',
		1,
		0
	)
	df['FTR-A'] = np.where(
		df['FTR'] == 'A',
		1,
		0
	)

	MATCH_RESULT_STAT_COLUMNS.append('FTR-0')
	MATCH_RESULT_STAT_COLUMNS.append('FTR-H')
	MATCH_RESULT_STAT_COLUMNS.append('FTR-D')
	MATCH_RESULT_STAT_COLUMNS.append('FTR-A')


if 'HTR' in COLUMNS:

	df['HTR-0'] = np.where(
		df['HTR'] == 'H',
		1,
		np.where(
			df['HTR'] == 'D',
			0,
			-1
		)
	)
	df['HTR-H'] = np.where(
		df['HTR'] == 'H',
		1,
		0
	)
	df['HTR-D'] = np.where(
		df['HTR'] == 'D',
		1,
		0
	)
	df['HTR-A'] = np.where(
		df['HTR'] == 'A',
		1,
		0
	)

	MATCH_RESULT_STAT_COLUMNS.append('HTR-0')
	MATCH_RESULT_STAT_COLUMNS.append('HTR-H')
	MATCH_RESULT_STAT_COLUMNS.append('HTR-D')
	MATCH_RESULT_STAT_COLUMNS.append('HTR-A')

df_match_id = get_match_data(df[['Date', 'Season_of_league', 'HomeTeam', 'AwayTeam'] + MATCH_RESULT_STAT_COLUMNS])
df_match_id = get_season_data(df_match_id)

df_match_id.fillna(0, inplace=True)
df_match_id.drop_duplicates(inplace=True)
df.drop_duplicates(inplace=True)
df = pd.merge(df, df_match_id.rename(columns={'team': 'HomeTeam'}), left_on=['Date', 'HomeTeam'], right_on=['Date', 'HomeTeam'], how='inner')

df.drop_duplicates(inplace=True)
df = pd.merge(df, df_match_id.rename(columns={'team': 'AwayTeam'}), left_on=['Date', 'AwayTeam'], right_on=['Date', 'AwayTeam'], suffixes=('_home', '_away'))
df.drop_duplicates(inplace=True)

if 'FTR-0' in MATCH_STAT_COLUMNS:

	df['point-rolling-3-diff'] = df['point-rolling-3_home'] - df['point-rolling-3_away']
	df['point-expanding-diff'] = df['point-expanding_home'] - df['point-expanding_away']

	MATCH_STAT_COLUMNS.append('point-rolling-3-diff')
	MATCH_STAT_COLUMNS.append('point-expanding-diff')


cols_id = ['index', 'Div', 'Date', 'HomeTeam', 'AwayTeam']


cols_result = ['FTR-H']

for odds in ODDS:

	if odds in COLUMNS:

		df[odds + '_prob'] = np.where(
			df[odds] == 0,
			0,
			1 / df[odds]
		)
		ODDS_COLUMNS.append(odds + '_prob')


# for index, row in df.iterrows():

for bookie in BOOKIES:

	if bookie + 'H' in COLUMNS:

		df[bookie + '_gmean'] = gmean(df[[bookie + x for x in ['H', 'D', 'A']]],axis=1) / 3
		df[bookie + '_std'] = df[[bookie + x for x in ['H', 'D', 'A']]].std(axis=1)

		ODDS_COLUMNS.extend([bookie + '_gmean', bookie + '_std'])

print(df.columns[df.isna().any()].tolist())

df = df.sort_values(by='Date').reset_index()
df.dropna(inplace=True)
df.to_csv(os.path.join(NORMALISED_DATA_PATH, 'normalised.txt'), sep='\t', index=False)

MATCH_STAT_COLUMNS.extend([col + '_home' for col in TEAM_STAT_COLUMNS])
MATCH_STAT_COLUMNS.extend([col + '_away' for col in TEAM_STAT_COLUMNS])
# MATCH_STAT_COLUMNS.extend(ODDS_COLUMNS)

# print(MATCH_STAT_COLUMNS)
# sys.exit()

with open(os.path.join(SETTINGS_PATH, 'normalised-x-columns.pkl'), 'wb') as f:
	pickle.dump(MATCH_STAT_COLUMNS, f)

with open(os.path.join(SETTINGS_PATH, 'normalised-odds-columns.pkl'), 'wb') as f:
	pickle.dump(ODDS_COLUMNS, f)

