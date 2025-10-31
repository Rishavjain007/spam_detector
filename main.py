from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, os, uvicorn

app = FastAPI(title="Spam Message Detector")

# CORS: for development allow all origins; lock down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static + templates setup (make sure these directories exist)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model and vectorizer
model_path = os.path.join("model", "spam_model.pkl")
vec_path = os.path.join("model", "vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vec_path):
    raise RuntimeError("Model files not found in 'model/' — run train_model.py first")

model = joblib.load(model_path)
vectorizer = joblib.load(vec_path)

class Message(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict_spam(msg: Message):
    # Defensive: ensure text is string
    txt = (msg.text or "").strip()
    text_vec = vectorizer.transform([txt])
    prob = float(model.predict_proba(text_vec)[0][1])  # spam probability
    pred = int(prob > 0.5)
    result = "🚨 Spam Message" if pred == 1 else "✅ Safe Message"
    # Return confidence as float percent (0-100)
    return {"prediction": pred, "result": result, "confidence": round(prob * 100, 2)}

# ✅ Auto-run the server (no need to type uvicorn command)
if __name__ == "__main__":
    try:
        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    except OSError:
        # If port 8000 is in use, automatically move to 8001
        print("⚠️ Port 8000 busy, trying port 8001...")
        uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
