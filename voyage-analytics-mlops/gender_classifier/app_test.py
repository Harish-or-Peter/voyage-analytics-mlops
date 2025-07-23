#app_test.py

import requests
import json


import mlflow.pyfunc

model = mlflow.pyfunc.load_model("mlflow_model")
print(model.metadata.signature)


# # Column names must exactly match what the model was trained with
# data = {
#     "columns": ["age", "name_length", "has_middle_name", "company_encoded"],
#     "data": [[25, 6, 1, 3]]
# }

# res = requests.post("http://127.0.0.1:5002/invocations", json=data)
# print("Status:", res.status_code)
# print("Response:", res.text)
