# 📰 News Sentiment Classifier (Task 5)

This project builds a *binary text classifier* that predicts whether a financial news sentence expresses a *positive* or *negative* sentiment. It uses a real-world dataset and saves the trained model for use in an API.

---

## 💡 How the Model Works

- *Dataset Used*: [descartes100/enhanced-financial-phrasebank](https://huggingface.co/datasets/descartes100/enhanced-financial-phrasebank) from Hugging Face
- The dataset contains sentences labeled as:
  - 0: Negative
  - 1: Neutral (filtered out)
  - 2: Positive

- We filter out *neutral* sentences and convert labels:
  - 0 → 0 (Negative)
  - 2 → 1 (Positive)

- *Preprocessing*: TF-IDF vectorization (TfidfVectorizer)
- *Model*: Logistic Regression (LogisticRegression)
- *Pipeline*: A Scikit-learn pipeline combines preprocessing and training

- *Final Accuracy*: ~92.4%

---

## 🧪 How to Run

1. *Clone the repository* or place the files in a directory
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Linux/Mac
   venv\Scripts\activate     # on Windows
   ```
3. Install dependencies:
     ```bash
    pip install -r requirements.txt
    ```
4. Run the training script:
    ```bash
    python news_sentiment_classifier.py
    ```

5. The trained model will be saved as:
    ```bash
    news_sentiment_classifier_model.joblib
    ```



---

📁 Files Included

File	Description

1. news_sentiment_classifier.py	- Python script that trains and saves model
2. news_sentiment_classifier_model.joblib	- Saved trained model (Joblib format)
3. README.md	- This documentation file
4. requirements.txt - The requirements file for instant install of dependancies as mentioned above


✅ Requirements

1. Python 3.8+
2. scikit-learn
3. datasets (Hugging Face)
4. joblib

