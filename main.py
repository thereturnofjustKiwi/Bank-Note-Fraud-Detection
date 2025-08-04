import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
from banknote_db import BankNote

app = FastAPI()

# CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bank-note-fraud-detection.onrender.com/"],  # For development only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open('classifier.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)

@app.get("/", response_class=HTMLResponse)
def get_home(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.post('/predict')
def predict(data: BankNote):
    data = data.model_dump()  # Use data.dict() if on Pydantic v1
    features = [[
        data['variance'],
        data['skewness'],
        data['curtosis'],
        data['entropy']
    ]]
    print("Predicting for:", features)
    prediction = classifier.predict(features)
    return {'prediction': prediction.tolist()}

@app.get('/about')
def about():
    return {'author': f'This page is made by Kevin'}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
