# HackathonProject: AI Warriors Against Political Misinformation

[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![AWS](https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Huggingface](https://img.shields.io/badge/Huggingface-FFD700?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

## Description
This project was developed during the Hackathon 2024 by Sustainista in Vienna, where it won first place in the Fake News Detection Challenge and received the additional award for "The Best Tech Solution."

### Challenge Description
**Title:** Combating Political Fake News with AI

**Background:** In the digital age, political misinformation and fake news have become rampant, influencing public opinion and potentially swaying election outcomes. The challenge was to develop an AI-powered solution to detect and mitigate the spread of fake news related to politics.

**Goal:** Create an innovative AI model that identifies, analyzes, and classifies political news content as legitimate or fake with high accuracy. The solution should incorporate real-time detection capabilities and be scalable across various digital platforms.

**Challenge Statement:** Participants were tasked with developing an AI system capable of discerning the veracity of political news stories, tweets, and other forms of digital content. The system should help social media platforms, news agencies, and end-users proactively combat the dissemination of false information.

## Solution
We first gathered datasets across the web and finally created a dataset with approx. 150.000 rows. We trained and fine-tuned the ROBERTA model from Hugging Face using PyTorch. We utilized Google Cloud GPU computational resources. After successfully training the model and achieving an accuracy score of 95%, we saved the model and deployed it using Flask API on AWS EC2 and AWS S3. We then created a user-friendly Telegram bot where users can input text and receive probabilities for fake and real news.

## Features
- **AI Model:** Utilizes the ROBERTA model fine-tuned for detecting fake news.
- **Deployment:** Flask API hosted on AWS EC2 and AWS S3 for serving the model.
- **User Interface:** Telegram bot that allows users to classify news as fake or real by inputting text.
- **Accuracy:** Achieved an accuracy score of 95% on the test dataset.

## Video Demo

Please find the demo of the project in the root of the project. "telegram_bot_demo.mp4"

## Usage
### Telegram Bot
Users can interact with the Telegram bot by sending text messages to classify them as fake or real news. The bot responds with probabilities for each classification. Attention! As you are reading this, telegram bot may not be available anymore, because it was hosted on temporary servers that were available during the challenge.

### API
The Flask API can be accessed to classify text programmatically. Send a POST request with the text data to the API endpoint to receive the classification probabilities.

## Analysis
An analysis of the model's accuracy, including testing results with precision, recall, and confusion matrix metrics, is provided in the `analysis` folder.

## Special Thanks
Special thanks to Hashym, my very good friend, for participating with me and contributing to this project! I appreciate your efforts! [@uulu](https://github.com/uulu)

## Acknowledgements
We would like to thank Sustainista for organizing the hackathon and providing the necessary resources and support.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
