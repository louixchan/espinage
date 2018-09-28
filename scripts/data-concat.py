import os
import pandas as pd


TARGET = 'P1'

DATA_PATH = f'../data/{TARGET}/raw'

NORMALISED_DATA_PATH = f'../data/{TARGET}/normalised'

FIXTURE_DATA_PATH = f'../data/fixtures'

DATE_FORMAT = '%d/%m/%y'

COLUMNS = []

with open(f'../data/{TARGET}/settings/raw-columns') as f:

	COLUMNS = f.read().split('\n')

	if COLUMNS[-1] == '':

		COLUMNS = COLUMNS[:-1]

RAW_FILES = os.listdir(DATA_PATH)

concat = None


for file in RAW_FILES:

	base = file[:2]

	if base[:2] != TARGET:
		continue

	print(file)

	data = pd.read_csv(os.path.join(DATA_PATH, file))

	data = data[COLUMNS]

	data['Date'] = pd.to_datetime(data['Date'], format=DATE_FORMAT)

	if concat is None:

		concat = data

	else:

		concat = concat.append(data)

if os.path.exists(os.path.join(FIXTURE_DATA_PATH, 'fixtures.csv')):

	data = pd.read_csv(os.path.join(FIXTURE_DATA_PATH, 'fixtures.csv'))

	data = data[data['Div'] == target][COLUMNS]

	data['Date'] = pd.to_datetime(data['Date'], format=DATE_FORMAT)

	if concat is None:

		concat = data

	else:

		concat = concat.append(data)

concat.to_csv(os.path.join(NORMALISED_DATA_PATH, 'data-concat.txt'), sep='\t', index=False)
