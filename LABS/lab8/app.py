from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

# Define the request body structure
class CommentRequest(BaseModel):
    reddit_comment: str

# Load the model at startup
@app.on_event('startup')
def load_model():
    global model_pipeline
    # Load the model directly using joblib as shown in the document
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")

# Root endpoint
@app.get('/')
def main():
    return {'message': 'Reddit comment classification model API'}

# Prediction endpoint
@app.post('/predict')
def predict(data: CommentRequest):
    # Prepare the data for prediction
    comment = [data.reddit_comment]
    
    # Make prediction
    predictions = model_pipeline.predict_proba(comment)
    
    return {'Predictions': predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)