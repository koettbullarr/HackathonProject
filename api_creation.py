import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch import argmax
from torch import no_grad
import json

# Load the model and tokenizer
MODEL_PATH = '/model/fine-tuned-model'  # Update with your model path
TOKENIZER_PATH = '/model/fine-tuned-tokenizer'  # Update with your tokenizer path

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Function to classify text
def classify(event, context):
    try:
        # Extract text from the request
        body = json.loads(event['body'])
        text = body['text']

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Make predictions
        with no_grad():
            outputs = model(**inputs)

        # Post-processing
        logits = outputs.logits
        prediction = 'fake' if argmax(logits, dim=-1).item() == 1 else 'real'
        probabilities = softmax(logits, dim=-1).cpu().numpy().tolist()[0]

        # Prepare response
        response = {
            "prediction": prediction,
            "probabilities": {
                "real": probabilities[0],
                "fake": probabilities[1]
            }
        }
        return {
            "statusCode": 200,
            "body": json.dumps(response)
        }
    except Exception as e:
        # Return error response
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
