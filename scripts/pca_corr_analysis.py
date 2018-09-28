import pandas as pd
import os
from sklearn.decomposition import PCA
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np

TARGET = 'P1'

SETTINGS_PATH = f'../data/{TARGET}/settings'

OUTPUT_PATH = f'../outputs/{TARGET}'

Y_COLUMN = 'FTR-0'

X_COLUMNS = []

PCA_COMPONENTS = 2

METHOD = 'CORR'

df = pd.read_csv(f'../data/{TARGET}/normalised/normalised.txt', delimiter='\t').reset_index(drop=True)


if METHOD == 'PCA':

	with open(os.path.join(SETTINGS_PATH, 'normalised-x-columns.pkl'), 'rb') as f:

		X_COLUMNS = pickle.load(f)

	df_x = df[X_COLUMNS]
	df_y = df[Y_COLUMN]

	print(df_x.columns[df_x.isna().any()].tolist())

	pca = PCA(n_components=PCA_COMPONENTS)
	pca.fit_transform(df_x, df_y)

	print(f"Explained Variance: {pca.explained_variance_ratio_}")

	pd.DataFrame(pca.components_,columns=df_x.columns,index = ['PC-1', 'PC-2']).transpose()\
		.to_csv(os.path.join(OUTPUT_PATH, 'pca.txt'), sep='\t')

elif METHOD == 'CORR':

	with open(os.path.join(SETTINGS_PATH, 'pca-x-columns'), 'r') as f:

		X_COLUMNS = f.read().split('\n')

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	cmap = cm.get_cmap('jet', 30)
	cax = ax1.imshow(df[X_COLUMNS].corr(), interpolation="nearest", cmap=cmap)
	ax1.grid(True)
	plt.title('Feature Correlation')
	labels = X_COLUMNS
	plt.yticks(np.arange(0.5, len(X_COLUMNS), 1) - 0.5, X_COLUMNS)
	fig.colorbar(cax, ticks=[-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
	plt.show()
