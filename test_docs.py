from fastapi.testclient import TestClient
import json
from src import main

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


less_sample = {"age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"}

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

if __name__ == "__main__":
    test_welcome_message()
    test_get_items()
    test_prediction_smaller()
    test_prediction_bigger()