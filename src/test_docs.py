from fastapi.testclient import TestClient
import json

try:
    import src.main as main
except:
    import main

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
    r = client.post("/prediction",data=json.dumps(more_sample))
    assert r.status_code == 200, r.json()
    assert str(r.json()) == ">50K"


less_sample = {
  "age": 25,
  "workclass": "Never-married",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Divorced",
  "occupation": "Adm-clerical",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "United-States"
}

def test_prediction_smaller():
    r = client.post("/prediction",data=json.dumps(less_sample))

    assert r.status_code == 200 , r.json()
    assert str(r.json()) == "<=50K"
    
def test_welcome_message():
    r = client.get("")

    assert r.status_code == 200, "test_welcome_message"
    assert r.json() == "Welcome to Project 4 for Salary Range Prediction!"


def test_get_items():
    r = client.get("/prediction")
    assert r.status_code != 200, "test_get_items"

# test_welcome_message()
# test_get_items()
# test_prediction_smaller()
# test_prediction_bigger()