from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# 建立 FastAPI 實例
app = FastAPI()

# 載入情感分析模型
sentiment_analysis = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

# 定義請求的資料格式
class TextData(BaseModel):
    text: str

# 建立一個 POST API 路由
@app.post("/predict")
def predict_sentiment(data: TextData):
    result = sentiment_analysis(data.text)
    return {"text": data.text, "result": result}
