from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

# 初始化 HuggingFace pipeline，指定使用 CPU
nlp = pipeline("sentiment-analysis", device=-1)  # -1 means using CPU

@app.post("/predict")
async def predict(text: str):
    result = nlp(text)
    return {"prediction": result}
