import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch import argmax
from torch import no_grad


model_path = './model/fine-tuned-model'
tokenizer_path = './model/fine-tuned-tokenizer'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def classify(text):
    # Tokenize the texts with truncation and padding
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Make predictions
    with no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = ['fake' if x == 1 else 'real' for x in argmax(logits, dim=-1).numpy()][0]
    probabilities = softmax(logits, dim=-1).cpu().numpy()[0]

    return prediction, probabilities


if __name__ == '__main__':
    while True:
        text = input('Input text: ')
        if text == "q":
            print("Breaking the loop")
            break
        prediction, probabilities = classify(text)
        print(f"Prediction: {prediction}\nProbability to be fake: {probabilities[1]}\nProbability to be real: {probabilities[0]}")



