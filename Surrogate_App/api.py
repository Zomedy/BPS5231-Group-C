"""
Data Center Heatwave Resilience - Web API & Frontend Host
Start command: uvicorn api:app --host 127.0.0.1 --port 8000 --reload
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from predictor import Surrogate, FEATURES, RANGES

app = FastAPI(
    title="Data Center Heatwave Resilience Surrogate",
    description="Surrogate inference engine for data center thermal response under heatwaves.",
    version="1.0.0"
)

# 载入冻结代理模型（保持原有逻辑不变）
model = Surrogate()

class Parameters(BaseModel):
    Heatwave_Intensity_C: float = Field(..., ge=0.0, le=10.0, description="热浪强度（增温）°C")
    Heatwave_Duration_day: int = Field(..., ge=1, le=5, description="持续天数（整数）1-5")
    IT_Load_Wm2: float = Field(..., ge=300.0, le=1000.0, description="机房IT负荷密度 W/m²")
    Capacity_Ratio: float = Field(..., ge=0.9, le=1.3, description="制冷设备额定容量裕量比")
    Cooling_Setpoint_C: float = Field(..., ge=20.0, le=27.0, description="末端空调送回风设定温度 °C")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": True,
        "features": FEATURES,
        "ranges": RANGES,
        "model_version": "stage9_frozen_stage8_classifier_stage7_regressor"
    }

@app.post("/predict")
def predict(data: Parameters):
    try:
        # 调用原始 predictor.py 进行真实推演
        return model.predict(data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

# 挂载静态前端资源 (提供 static/index.html 网页访问)
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(static_dir / "index.html"))