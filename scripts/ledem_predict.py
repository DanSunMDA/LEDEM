#!/usr/bin/env python
import argparse
import textwrap
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

### Usage
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
		description=textwrap.dedent('''\
		--------------------------------
			Whole-genome prediction
		--------------------------------
		'''))
parser.add_argument('-i', help='The input CSV file prepared by preprocessTrain', required=True)
parser.add_argument('-m', help='A trained XGBoost model (stored as an joblib object)', required=True)
parser.add_argument('-s', help='The scaler used in the XGBoost model (stored as an joblib object)', required=True)
parser.add_argument('-o', help='Output directory', required=True)
args = parser.parse_args()

## Read data
model = joblib.load(args.m)
scaler = joblib.load(args.s)
X0 = genfromtxt(args.i, delimiter=',')
## Standardize data
X = scaler.transform(X0)
## Prediction
y_pred = model.predict(X)
y_prob = model.predict_proba(X)
np.savetxt(args.o + '/scores0.txt', y_prob)
