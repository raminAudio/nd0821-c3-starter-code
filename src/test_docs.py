from fastapi.testclient import TestClient
import json
import src.main as main
# from src import main

client = TestClient(main.app)


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
    print(r.json())

    assert r.status_code == 200, r.json()
    assert str(r.json()) == ">50K"


less_sample = {'age': 10,
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
 'hours-per-week': 10,
 'native-country': 'United-States'}


def test_prediction_smaller():
    r = client.post("/predict",data=json.dumps(less_sample))
    print(r.json())

    assert r.status_code == 200 , r.json()
    assert str(r.json()) == "<=50K"
    
def test_welcome_message():
    r = client.get("")
    print(r.json())

    assert r.status_code == 200, "test_welcome_message"
    assert r.json() == "Welcome to Project 4 for Salary Range Prediction!"


def test_get_items():
    r = client.get("/predict")
    assert r.status_code != 200, "test_get_items"

test_welcome_message()
test_get_items()
test_prediction_smaller()
test_prediction_bigger()