from fastapi.testclient import TestClient
import json
from src import main

client = TestClient(main.app)

print(client)

more_sample = {'age': 50,
 'workclass': 'Private',
 'fnlgt': 154374,
 'education': 'Bachelor',
 'education-num': 13,
 'marital-status': 'Married-civ-spouse',
 'occupation': 'Tech-support',
 'relationship': 'Husband',
 'race': 'White',
 'sex': 'Male',
 'capital-gain': 0,
 'capital-loss': 0,
 'hours-per-week': 40,
 'native-country': 'United-States'}


def test_prediction_bigger():
    r = client.post("/predict",data=json.dumps(more_sample))
    assert r.status_code == 200, r.status_code
    assert str(r.json()) == ">50K"


less_sample = {'age': 80,
                'workclass': 'Private',
                'fnlgt': 154374,
                'education': 'HS-grad',
                'education-num': 2,
                'marital-status': 'Married-civ-spouse',
                'occupation': 'Tech-support',
                'relationship': 'Husband',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 0,
                'capital-loss': 0,
                'hours-per-week': 20,
                'native-country': 'United-States'}

def test_prediction_smaller():
    r = client.post("/predict",data=json.dumps(less_sample))
    assert r.status_code == 200 , r.status_code
    assert str(r.json()) == "<=50K"
    
def test_welcome_message():
    r = client.get("")
    assert r.status_code == 200, "test_welcome_message"
    assert r.json() == "Welcome to Project 4 for Salary Range Prediction!"


def test_get_items():
    r = client.get("/predict")
    assert r.status_code != 200, "test_get_items"

    