import requests
import pandas as pd
from tqdm import tqdm

test_data = pd.read_csv("data/test.csv")
test_data = test_data.drop('row ID', axis=1)

# Iterate over each row in the test data, convert it to JSON, and call the API
yes_count = 0
total = 0

for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
    json_data = row.to_json()
    response = requests.post("http://127.0.0.1:8000/predict", data=json_data)
    prediction = response.json()
    yes_count += prediction['prediction']
    total += 1

print(f"Total: {total}\tYes: {yes_count}\tNo: {total - yes_count}")
