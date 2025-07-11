from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load saved model
model = joblib.load("news_sentiment_classifier_model.joblib")

app = FastAPI()

# Fix CORS error when calling from HTML
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body format
class NewsInput(BaseModel):
    text: str

# Prediction route
@app.post("/predict")
async def predict_sentiment(input: NewsInput):
    prediction = str(model.predict([input.text])[0] ) # 0 or 1
    label = "negative" if prediction == 0 else "positive" 
    return {"label": prediction}