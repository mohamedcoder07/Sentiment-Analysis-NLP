import joblib
import warnings
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
warnings.filterwarnings("ignore")

from src.preprocessing.vectorizer import WordVectorizer



app = FastAPI(title= "Amazon Sentiment Analysis API")


class Review(BaseModel):
    text: str
    sentiment: str


BASE_DIR = Path.cwd()
model_path =  BASE_DIR / "saved_models" / "model_lightgbm.pkl"
classifier = joblib.load(model_path)
vectorizer = WordVectorizer(method = "transformers")

print(model_path)

@app.get("/")
def data_description():
    return ""



@app.post("/predict-sentiment", response_model=str)
async def predict_sentiment(review: str):
    embed = vectorizer.get_embeddings(review)
    prediction = classifier.predict(embed)[0]
    return prediction




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)