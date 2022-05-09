import pandas as pd
import requests
import json

input_ = {"age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"}

url = 'https://raminapi.herokuapp.com/predict/'


r = requests.post(url, json=input_)
print("Response code: ", r.status_code)
print("Response body: ", r.json())
assert r.status_code == 200