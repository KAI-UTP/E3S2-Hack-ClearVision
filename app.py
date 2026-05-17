import os
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from detector import SceneDetector

app = FastAPI(title="ClearVision")
templates = Jinja2Templates(directory="templates")

# Single-worker executor — YOLO is not thread-safe for parallel calls
_executor = ThreadPoolExecutor(max_workers=1)

_detector: SceneDetector | None = None
_detector_lock = threading.Lock()
_load_error: str | None = None


def _load_detector() -> SceneDetector:
    """Thread-safe model loader. Safe to call from multiple threads."""
    global _detector, _load_error
    with _detector_lock:
        if _detector is None and _load_error is None:
            try:
                _detector = SceneDetector()
            except Exception as exc:
                _load_error = str(exc)
                raise
    return _detector


def _run_detect(image_bytes: bytes) -> dict:
    d = _load_detector()
    return d.detect(image_bytes)


# Kick off model loading immediately in a background thread so it is
# ready (or nearly ready) before the first user request arrives.
threading.Thread(target=_load_detector, daemon=True, name="model-warmup").start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(frame: UploadFile = File(...)):
    if _load_error:
        return JSONResponse({"error": _load_error}, status_code=500)
    image_bytes = await frame.read()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run_detect, image_bytes)
    return JSONResponse(result)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": _detector is not None,
        "model_error": _load_error,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
