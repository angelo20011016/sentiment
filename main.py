from fastapi import FastAPI, Body
from transformers import pipeline

app = FastAPI()

# 初始化 HuggingFace pipeline，指定使用 CPU
nlp = pipeline("sentiment-analysis", device=-1)  # -1 means using CPU

@app.post("/predict")
async def predict(text: str = Body(...)):  # 使用 Body(...) 指定從 body 中接收
    result = nlp(text)
    return {"prediction": result}
