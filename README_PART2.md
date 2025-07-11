# 📰 News Sentiment Classifier (Task 6)

This is a simple machine learning project that classifies news sentence descriptions into **positive** or **negative** sentiment using a Logistic Regression model trained on financial news data.

---

## 🔍 How It Works

- We use the [`descartes100/enhanced-financial-phrasebank`](https://huggingface.co/datasets/descartes100/enhanced-financial-phrasebank) dataset, which contains news phrases labeled as:
  - `0`: Negative  
  - `1`: Neutral  
  - `2`: Positive

- We filter out the **neutral** class and map:
  - `0` → **Negative**  
  - `2` → **Positive**

- The model uses:
  - `TfidfVectorizer()` to convert text into numerical features
  - `LogisticRegression()` as the classifier
  - A `Pipeline()` to combine both steps for easy training and prediction

- Achieves ~**93.5% accuracy** on the financial dataset

---

## ⚠️ Important Note

> This model is trained **only on financial news data**.
>
> It is **not designed to handle general sentiment** (e.g., movie reviews, weather, daily speech).
>
> Example:
> - ✅ `"The company faced massive losses"` → predicted as negative  
> - ❌ `"The movie was boring"` → incorrectly predicted as positive

---

## 📁 Files

| File Name                             | Description                                 |
|--------------------------------------|---------------------------------------------|
| `news_sentiment_classifier.py`       | Script that trains and saves the model      |
| `news_sentiment_classifier_model.joblib` | Saved model using joblib                 |
| `main.py`                            | FastAPI backend API                         |
| `index.html`                         | Frontend to input sentence and show result  |
| `README.md`                          | Project explanation (this file)             |

---

## 🚀 How to Run the Project

### 1️⃣ Train the Model (if not already trained)

```python
python news_sentiment_classifier.py
```

This saves the model as `news_sentiment_classifier_model.joblib`.

---

### 2️⃣ Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

The API will be available at:  
[http://127.0.0.1:8000](http://127.0.0.1:8000)

---

### 3️⃣ Use the Frontend

- Open `index.html` in your browser
- Type a financial news sentence like:

```text
The company reported strong Q2 earnings.
```

- Click **Predict Sentiment**
- The prediction result (positive/negative) will appear below

---

## 🔄 API Format

### `POST /predict`

**Request:**

```json
{
  "text": "The company faced a major loss this year."
}
```

**Response:**

```json
{
  "label": "negative"
}
```

---

## ✅ Example Predictions

| Input Sentence                                         | Expected Output |
|--------------------------------------------------------|-----------------|
| The market showed strong recovery signs                | positive        |
| The company faced huge losses this year                | negative        |
| The sky is blue                                        | (likely) positive |
| The movie was boring and badly written                 | (wrongly) positive |
| asdjkhqwieu zxczxc                                     | positive (default class) |

---

## 🙌 Credits

- **Dataset**: [`descartes100/enhanced-financial-phrasebank`](https://huggingface.co/datasets/descartes100/enhanced-financial-phrasebank)  
- **Tools Used**: FastAPI, scikit-learn, joblib, HTML, JavaScript
