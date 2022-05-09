# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from ml.evaluate_model import performance
import pickle

# Add code to load in the data.
data = pd.read_csv('data/census_clean.csv')
data = data.drop(["Unnamed: 0"],axis=1)
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Save encoded data 
pickle.dump(X_train, open('data/xTrain.pickle','wb'))
pickle.dump(encoder, open('data/encoder.pickle','wb'))
pickle.dump(lb, open('data/lb.pickle','wb'))

# Train and save a model.
model = train_model(X_train,y_train)

# Evaluating the model performance on slices of the test dataset 
evaluation = performance(model, cat_features, test, encoder, lb)