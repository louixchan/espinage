import tensorflow as tf
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime
import pandas as pd
import numpy as np
import os

NAME = 'P1'

DATA_PATH = f'../data/{NAME}/normalised'

SETTINGS_PATH = f'../data/{NAME}/settings'

MODELS_PATH = f'../models/{NAME}'

OUTPUTS_PATH = f'../outputs/{NAME}'

X_COLUMNS = []

Y_COLUMNS = ['FTR-0']

ID_COLUMNS = ['Date', 'HomeTeam', 'AwayTeam']

TRAIN_TEST_CUTOFF_DATE = datetime(2017, 7, 1)

MODEL_SETTINGS = [
	{
		'type': 'Dense',
		'shape': None,
		'neurons': 27
	},
	{
		'type': 'BatchNormalization'
	},
	{
		'type': 'Dense',
		'neurons': 243,
		'activation': 'relu'
	},
	{
		'type': 'Dropout',
		'rate': 0.1
	},
	{
		'type': 'Dense',
		'neurons': 27,
		'activation': 'relu'
	},
	{
		'type': 'Dropout',
		'rate': 0.1
	},
	{
		'type': 'Dense',
		'neurons': 3,
		'activation': 'softmax'
	}
]

MODEL_COMPILATION_SETTINGS = {
	'loss': 'sparse_categorical_crossentropy',
	'optimizer': Adam(lr=0.001, decay=1e-6),
	'metrics': ['accuracy']
}

MODEL_EPOCH = 2

MODEL_BATCH_SIZE = 32

def configure_model(settings=MODEL_SETTINGS):
	
	if not type(settings) == list:
		
		settings = list(settings)
		
	model = Sequential()
	
	input_layer = True
	
	for layer in settings:
		
		if input_layer:
			
			model.add(Dense(
				layer['neurons'],
				input_shape=layer['shape']
			))
			
			input_layer = False
			
		else:
			
			if layer['type'] == 'Dense':
				
				model.add(Dense(
					layer['neurons'],
					activation=layer['activation'],
					kernel_initializer=RandomNormal()
				))
				
			elif layer['type'] == 'Dropout':
				
				model.add(Dropout(layer['rate']))
				
			elif layer['type'] == 'BatchNormalization':
				
				model.add(BatchNormalization())
			
			# Compile model
	model.compile(
		# MODEL_COMPILATION_SETTINGS
		loss=MODEL_COMPILATION_SETTINGS['loss'],
		optimizer=MODEL_COMPILATION_SETTINGS['optimizer'],
		metrics=MODEL_COMPILATION_SETTINGS['metrics']
	)
	
	return model
	
	
def load_data(val_rate=0.1, x_cols=X_COLUMNS, y_cols=Y_COLUMNS, id_cols=ID_COLUMNS):
	
	global X_COLUMNS
	
	df = pd.read_csv(os.path.join(DATA_PATH, 'normalised.txt'), delimiter='\t')
	
	df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
	
	df['FTR-0'] = df['FTR-0'] + 1
	
	if len(x_cols) == 0:
	
		with open(os.path.join(SETTINGS_PATH, 'pca-x-columns'), 'r') as f:
			
			X_COLUMNS = f.read().split('\n')
			
			X_COLUMNS = X_COLUMNS[:-1] if X_COLUMNS[-1] == '' else X_COLUMNS
			
			x_cols = X_COLUMNS
			
	df = df[id_cols + y_cols + x_cols]
	
	df_train = df[df['Date'] <= TRAIN_TEST_CUTOFF_DATE]
	df_train.sample(frac=1)
	df_validation = df_train[:int(len(df_train) * val_rate)]
	df_train = df_train[int(len(df_train) * val_rate):]
	df_test = df[df['Date'] > TRAIN_TEST_CUTOFF_DATE]
	
	print(df_train[X_COLUMNS].values.shape)
	print(df_train[Y_COLUMNS].values.shape)
	print(df_validation[X_COLUMNS].values.shape)
	print(df_validation[Y_COLUMNS].values.shape)
	print(df_test[X_COLUMNS].values.shape)
	print(df_test[Y_COLUMNS].values.shape)
	
	return \
		df_train[ID_COLUMNS].values, df_train[X_COLUMNS].values, df_train[Y_COLUMNS].values, \
		df_validation[ID_COLUMNS].values, df_validation[X_COLUMNS].values, df_validation[Y_COLUMNS].values, \
		df_test[ID_COLUMNS].values, df_test[X_COLUMNS].values, df_test[Y_COLUMNS].values
	
	
def main():
	
	tensorboard = TensorBoard(log_dir=f"../logs/{NAME}")
	
	train_id, train_x, train_y, validation_id, validation_x, validation_y, test_id, test_x, test_y = load_data()
	
	MODEL_SETTINGS[0]['shape'] = train_x.shape[1:]
	print(MODEL_SETTINGS[0]['shape'])
	
	losses = []
	accuracies = []
	
	for i in range(40):
		
		starttime = datetime.now()
		
		os.mkdir(os.path.join(MODELS_PATH, f'{starttime}'))
		
		filepath = "NN-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
		checkpoint = ModelCheckpoint(
			os.path.join(MODELS_PATH, f'{starttime}',
			             "{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
			                               mode='max')))  # saves only the best ones
	
		model = configure_model()
		
		print(model.summary())
	
		# Train model
		history = model.fit(
			train_x, train_y,
			batch_size=MODEL_BATCH_SIZE,
			epochs=MODEL_EPOCH,
			validation_data=(validation_x, validation_y),
			callbacks=[tensorboard, checkpoint],
		)
		
		# # Score model
		score = model.evaluate(test_x, test_y, verbose=0)
		print('Test loss:', score[0])
		print('Test accuracy:', score[1])
		
		losses.append(score[0])
		accuracies.append(score[1])
		
		# Save model
		model.save(os.path.join(MODELS_PATH, f'{starttime}/NN-Final.model'))
	
	with open(os.path.join(OUTPUTS_PATH, 'losses.txt'), 'w+') as f:
		
		f.write('\n'.join([str(x) for x in losses]))
	
	with open(os.path.join(OUTPUTS_PATH, 'accuracies.txt'), 'w+') as f:
		
		f.write('\n'.join([str(x) for x in accuracies]))
		
	print(np.mean(losses), np.std(losses))
	print(np.mean(accuracies), np.std(accuracies))
	
	
if __name__ == '__main__':
	
	main()