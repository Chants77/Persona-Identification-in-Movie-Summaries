import csv
import json
from transformers import pipeline
import torch
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def parse_response(response_text):
    try:
        data = json.loads(response_text)
        # Optionally, you can add further processing here if needed
        return data
    except json.JSONDecodeError:
        # Handle JSON parsing error
        print("Failed to parse JSON response.")
        return None


generated_text = {'role': 'assistant', 'content': 'absent_minded_professor'}
response = parse_response(generated_text)
print(response)
