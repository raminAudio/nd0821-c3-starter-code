import pytest
import pandas as pd
import tensorflow as tf
import pickle
import logging
import os 
print(os.path.abspath(os.pardir))
ROOT_DIR = os.path.abspath(os.curdir)

@pytest.fixture
def dataTest():
    """ fixture for input dataset """
    df = pd.read_csv( os.path.join(ROOT_DIR,'data/census_clean.csv'))
    return df

def test_data_shape(dataTest):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert dataTest.shape == dataTest.dropna().shape, "Dropping null changes shape."

def test_model_exist():
    """ does model exist? """
    model = tf.keras.models.load_model( os.path.join(ROOT_DIR,'model/TFmodel_v1.h5'))
    assert model != None, "Model does not exist"

def test_model_predicts():
    try: 
        model = tf.keras.models.load_model( os.path.join(ROOT_DIR,'model/TFmodel_v1.h5'))
        encoded_data = pickle.load(open( os.path.join(ROOT_DIR,'data/xTrain.pickle'),'rb'))
        model.predict(encoded_data)
    except: 
        assert False, "Model could not predict on encoded data"

        
def test_model_prediction_shape(dataTest):
    """ Test model can predict """
    try: 
        model = tf.keras.models.load_model( os.path.join(ROOT_DIR,'model/TFmodel_v1.h5'))
        encoded_data = pickle.load(open( os.path.join(ROOT_DIR,'data/xTrain.pickle'),'rb'))
        prediction = model.predict(encoded_data)
        assert prediction.shape[1] == 2
    except: 
        assert False
        
# def test_yaml():
#     try:
#         with open("./params.yaml", "rb") as f:
#             params = yaml.load(f, Loader=Loader)
#     except: 
#         assert False, "Could not find hyperparameters to train the model"
    