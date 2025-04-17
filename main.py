from fastapi import FastAPI, Body, HTTPException
from transformers import pipeline
import logging

# 配置日誌記錄
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 初始化 HuggingFace pipeline，指定使用特定模型
try:
    logger.info("正在載入情緒分析模型...")
    nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    logger.info("模型載入成功!")
except Exception as e:
    logger.error(f"模型載入失敗: {e}")
    nlp = None

@app.post("/predict")
async def predict(text: str = Body(...)):
    if nlp is None:
        raise HTTPException(status_code=503, detail="模型尚未準備好，請稍後再試")
    
    try:
        result = nlp(text)
        return {"prediction": result}
    except Exception as e:
        logger.error(f"預測時發生錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"預測失敗: {str(e)}")

@app.get("/")
async def read_root():
    model_status = "已載入" if nlp is not None else "未載入"
    return {"message": f"API 正在運行! 模型狀態: {model_status}"}