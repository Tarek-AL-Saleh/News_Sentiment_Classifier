# 📰 News Sentiment Classifier (Binary)

This project classifies financial news sentences into **Positive** or **Negative** sentiment using a machine learning model trained with scikit-learn. It covers both:

- ✅ Task 5: Model Training and Saving
- ✅ Task 6: Live Prediction API & Frontend

---

## 🔍 How the Model Works

- We use the [`descartes100/enhanced-financial-phrasebank`](https://huggingface.co/datasets/descartes100/enhanced-financial-phrasebank) dataset which has 3 classes:
  - `0` → Negative  
  - `1` → Neutral (we discard this)  
  - `2` → Positive (we convert this to `1`)

- We train a **Pipeline** consisting of:
  - `TfidfVectorizer()` → Converts text into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
  - `LogisticRegression()` → A simple but powerful linear model for binary classification.

- Model achieves approximately **93.5% accuracy** on the test set.

⚠️ This classifier is trained specifically on financial news data. It does not perform well on unrelated topics like movies, weather, or general conversation.

---

## ✅ Which Method We Used and Why?

We used:

- ➡️ Model: `LogisticRegression`  
- ➡️ Preprocessing: `TfidfVectorizer`  
- ➡️ Tool for saving: `joblib` → `news_sentiment_classifier_model.joblib`

🔹 Why Logistic Regression?  
It is simple, interpretable, fast to train, and effective for binary text classification.

🔹 Why TF-IDF?  
It gives importance to words that are frequent in a sentence but rare across the dataset, making it well-suited for sentiment classification.

🔹 Why joblib?  
We chose `joblib` over `pickle` because it handles large NumPy arrays and scikit-learn objects more efficiently. It’s also faster when saving and loading models that involve vectorizers.

---

## 📁 Project Files

| File                                     | Purpose                                   |
|------------------------------------------|-------------------------------------------|
| `news_sentiment_classifier.py`           | Trains the model and saves it with joblib |
| `news_sentiment_classifier_model.joblib` | Saved ML model (TF-IDF + LogisticRegression) |
| `main.py`                                | FastAPI backend to serve predictions      |
| `index.html`                             | Simple frontend to call the API           |
| `README.md`                              | Documentation (this file)                 |

---

## 🚀 How to Run the Project

### 1️⃣ Train the Model (Task 5)

```bash
python news_sentiment_classifier.py
```

➡️ This saves `news_sentiment_classifier_model.joblib`.

---

### 2️⃣ Start the FastAPI Backend (Task 6)

```bash
uvicorn main:app --reload
```

➡️ Access the API at:  
http://127.0.0.1:8000

---

### 3️⃣ Use the Frontend (Task 6)

- Open `index.html` in your browser
- Type a financial news sentence (Example:  
  `"The company reported record quarterly profits."`)
- Click “Predict Sentiment”
- The result will display below as either `positive` or `negative`

---

## 🔄 API Details

### `POST /predict`

- Request:

```json
{
  "text": "The company faced significant losses this year."
}
```

- Response:

```json
{
  "label": "negative"
}
```

---

## ✅ Example Predictions

| Input Sentence                                | Prediction |
|----------------------------------------------|------------|
| The market showed strong recovery signs      | positive   |
| The company faced huge losses this year      | negative   |
| The sky is blue                              | (likely) positive |
| The movie was poorly received by critics     | (likely) positive |
| asdkjqw qweioqwe qweqw                       | positive (default) |

⚠️ Note: The model is accurate for financial sentences but unreliable for unrelated topics like movies or random text.

---

## 🙌 Tools & Credits

- Dataset: [`descartes100/enhanced-financial-phrasebank`](https://huggingface.co/datasets/descartes100/enhanced-financial-phrasebank)
- Libraries: FastAPI, scikit-learn, joblib, HTML, JavaScript

---