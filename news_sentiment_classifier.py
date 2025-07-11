from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import joblib

raw_dataset=load_dataset("descartes100/enhanced-financial-phrasebank")
dataset=raw_dataset["train"]["train"]
print(dataset[0])

X=[]
Y=[]
for entry in dataset:
    if entry["label"]==1:
        continue
    X.append(entry["sentence"])
    Y.append(0 if entry["label"]==0 else 1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
pipeline=Pipeline([
    ("tfidf",TfidfVectorizer()),
    ("lg",LogisticRegression(max_iter=200))
])
pipeline.fit(x_train,y_train)
y_predict= pipeline.predict(x_test)

print(f"Report: {classification_report(y_test,y_predict)}")
print(f"Accuracy: {accuracy_score(y_test,y_predict)}")

joblib.dump(pipeline,"news_sentiment_classifier_model.joblib")
