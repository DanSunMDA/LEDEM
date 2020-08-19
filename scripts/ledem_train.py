#!/usr/bin/env python
import argparse
import textwrap
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
import joblib

### Functions
def convertType(s):
	if s == 'None':
		return None
	try:
		return int(s)
	except ValueError:
		try:
			return float(s)
		except ValueError:
			try:
				return bool(s)
			except ValueError:
				return s

def listToDict(l, delimiter='='):
	aDict = {}
	for e in l:
		e_split = e.split(delimiter)
		aDict[e_split[0]] = convertType(e_split[1])
	return aDict

### Main
if __name__ == '__main__':
	### Usage
	parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
			description=textwrap.dedent('''\
			--------------------------------
			    Trains an XGBoost model
			--------------------------------
			'''))
	parser.add_argument('-i', help='The input CSV file prepared by preprocessTrain', required=True)
	parser.add_argument('-x', help='Parameters to pass to XGBClassifier', required=True)
	parser.add_argument('-t', help='If set, performs the hyperparameter tuning', action='store_true')
	parser.add_argument('-n_iter', type=int, help='n_iter for hyperparameter tuning', required=False)
	parser.add_argument('-n_jobs', type=int, help='n_jobs for hyperparameter tuning', required=False)
	parser.add_argument('-o', help='Output directory', required=True)
	args = parser.parse_args()
	args.x = args.x.split(',')

	### Read training dataset
	data = genfromtxt(args.i, delimiter=',')
	X_train0 = data[:,1:]
	y_train = np.int64(data[:,0])

	### Standardize data
	scaler = StandardScaler().fit(X_train0)
	X_train = scaler.transform(X_train0)

	### Train an XGBoost model
	if args.t: # If -t is set, performs a random grid Search using five-fold cross-validation to achieve optimized hyperparameters
		print('Hyperparameter tuning is enabled (n_iter=', args.n_iter, ', n_jobs=', args.n_jobs, ')', sep='')
		if not args.x == ['']:
			params = listToDict(args.x)
			if 'n_jobs' in params:
				model0 = XGBClassifier(n_jobs=params['n_jobs'])
			else:
				model0 = XGBClassifier()
		else:
			model0 = XGBClassifier()
		param_space = {'n_estimators': [50,100,150,200,250,300,350,400,450,500],
				'learning_rate': [0.05,0.10,0.15,0.20,0.25,0.30],
				'max_depth': [3,4,5,6,8,10,12,15],
				'min_child_weight':[1,3,5,7],
				'gamma': [0.0,0.1,0.2,0.3,0.4],
				'colsample_bytree':[0.3,0.4,0.5,0.7]}
		rsearch = RandomizedSearchCV(estimator=model0, param_distributions=param_space, n_iter=args.n_iter, n_jobs=args.n_jobs, verbose=2, random_state=21)
		model = rsearch.fit(X_train, y_train)
		print('Tuned hyperparameters:', rsearch.best_params_)
	else: # If -t is not set, uses default settings
		print('Hyperparameter tuning is not enabled.')
		if not args.x == ['']:
			print('Using user-defined hyperparameters ', end=' ')
			params = listToDict(args.x)
			print(params)
			model = XGBClassifier(**params)
		else:
			print('Using default hyperparameters..')
			model = XGBClassifier()
		model.fit(X_train, y_train)
		print(model)

	### Save the model and scaler to disk
	joblib.dump(model, open(args.o + '/model.joblib.dat', 'wb')) 
	joblib.dump(scaler, open(args.o + '/scaler.joblib.dat', 'wb')) 
