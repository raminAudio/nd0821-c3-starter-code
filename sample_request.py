import pandas as pd
import requests
import json

input_ = {"age": 31,
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


print("Sample Request GET")

url = 'http://127.0.0.1:8000/'# local testing get
# url = "http://0.0.0.0:5000" #live api get test

r = requests.get(url)
print(r.json())
assert r.status_code == 200

print("----------------------")
print("Sample Request POST")

url = 'http://127.0.0.1:8000/predict/' # local testing post
# url = "http://0.0.0.0:5000/predict/" #live api post test

r = requests.post(url,data=json.dumps(input_))
assert r.status_code == 200
print(r.json())

