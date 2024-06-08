import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch import argmax, no_grad
import re
from flask import Flask, request, jsonify

# Load the model and tokenizer
MODEL_PATH = './model/fine-tuned-model'  # Update with your model path
TOKENIZER_PATH = './model/fine-tuned-tokenizer'  # Update with your tokenizer path

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Create a Flask app
app = Flask(__name__)


# Function to replace whitespace with a single space
def replace_whitespace_with_space(text):
    return re.sub(r'\s+', ' ', text).strip()


# Function to classify text
@app.route('/classify', methods=['POST'])
def classify():
    try:
        # Extract text from the request
        data = request.json
        text = data['text']

        # Preprocess text
        processed_text = replace_whitespace_with_space(text)

        # Tokenize the text
        inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

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
        return jsonify(response), 200
    except Exception as e:
        # Return error response
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
