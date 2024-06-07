import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


model_path = './model/fine-tuned-model'
tokenizer_path = './model/fine-tuned-tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def classify(text):
    # Tokenize the texts with truncation and padding
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = ['fake' if x == 1 else 'real' for x in torch.argmax(logits, dim=-1).numpy()][0]
    probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

    return prediction, probabilities


if __name__ == '__main__':
    while True:
        text = input('Input text: ')
        if text == "q":
            print("Breaking the loop")
            break
        prediction, probabilities = classify(text)
        print(f"Prediction: {prediction}\nProbability to be fake: {probabilities[1]}\nProbability to be real: {probabilities[0]}")



