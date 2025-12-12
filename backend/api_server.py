"""
FastAPI server for emotion detection via Ollama.

Optimized for lower latency:
- Prompt rút gọn, num_predict giảm (32).
- Sử dụng stream=True tới Ollama, dừng sớm khi đã trích xuất nhãn.
- Warm-up lúc khởi động để giữ nóng model.
- Reuse HTTP session để giảm chi phí kết nối.
"""

import json
import re
from typing import Dict, Generator, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gpt-oss:20b"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MODEL_OPTIONS = {"temperature": 0, "num_predict": 32}

# Reuse session to avoid TCP/TLS handshake overhead
session = requests.Session()


def build_prompt(text: str) -> str:
    return (
        "Classify the emotion of the sentence. Allowed labels: sadness, joy, love, anger, fear, surprise.\n"
        "Reply ONLY JSON: {\"label\": \"<label>\"}.\n"
        f"Sentence: {text}"
    )


def normalize_label(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z ]", " ", raw).strip().lower()
    if not cleaned:
        return ""

    candidate = cleaned.split()[0]
    synonyms: Dict[str, str] = {
        "happy": "joy",
        "happiness": "joy",
        "joyful": "joy",
        "ecstatic": "joy",
        "sad": "sadness",
        "depressed": "sadness",
        "angry": "anger",
        "mad": "anger",
        "furious": "anger",
        "afraid": "fear",
        "scared": "fear",
        "fearful": "fear",
        "terrified": "fear",
        "surprised": "surprise",
        "shocked": "surprise",
        "astonished": "surprise",
        "love": "love",
        "loved": "love",
        "loving": "love",
    }
    if candidate in LABELS:
        return candidate
    return synonyms.get(candidate, candidate)


def extract_label_from_json(text: str) -> str:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return ""
    try:
        parsed = json.loads(match.group(0))
        return str(parsed.get("label", ""))
    except Exception:
        return ""


def post_ollama_stream(prompt: str) -> requests.Response:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": MODEL_OPTIONS,
    }
    response = session.post(API_URL, json=payload, stream=True, timeout=180)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error: {response.text}")
    return response


def classify(text: str, early_stop: bool = True) -> Dict[str, Optional[str]]:
    """
    Streaming từ Ollama và dừng sớm ngay khi trích xuất được nhãn.
    """
    prompt = build_prompt(text)
    response = post_ollama_stream(prompt)

    buffer = ""
    done_reason: Optional[str] = None

    try:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            done_reason = chunk.get("done_reason") or done_reason
            piece = chunk.get("response", "")
            if piece:
                buffer += piece

            label_candidate = normalize_label(
                extract_label_from_json(buffer) or buffer
            )

            if early_stop and label_candidate:
                # Đã đủ thông tin -> dừng stream để giảm latency
                return {
                    "predicted_emotion": label_candidate,
                    "done_reason": done_reason or "stopped_early",
                }

            if chunk.get("done"):
                final_label = label_candidate
                return {
                    "predicted_emotion": final_label,
                    "done_reason": done_reason or "done",
                }
    finally:
        response.close()

    raise HTTPException(status_code=502, detail="No response from Ollama stream")


def stream_events(text: str) -> Generator[str, None, None]:
    """
    SSE generator: stream token chunks ra client, kết thúc bằng event 'done'
    kèm nhãn đã chuẩn hóa.
    """
    prompt = build_prompt(text)
    response = post_ollama_stream(prompt)

    buffer = ""
    done_reason: Optional[str] = None

    try:
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            done_reason = chunk.get("done_reason") or done_reason
            piece = chunk.get("response", "")

            if piece:
                buffer += piece
                yield f"data: {json.dumps({'token': piece})}\n\n"

            if chunk.get("done"):
                label = normalize_label(extract_label_from_json(buffer) or buffer)
                payload = {
                    "sentence": text,
                    "predicted_emotion": label,
                    "done_reason": done_reason or "done",
                }
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                break
    finally:
        response.close()


class ClassifyRequest(BaseModel):
    sentence: str


class ClassifyResponse(BaseModel):
    sentence: str
    predicted_emotion: Optional[str]
    done_reason: Optional[str]


app = FastAPI(title="Emotion Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def warm_up() -> None:
    """
    Gửi request warm-up để tải model vào bộ nhớ, tránh chậm ở lần đầu.
    """
    try:
        classify("warmup", early_stop=True)
    except Exception:
        # Bỏ qua lỗi warm-up để không chặn server khởi động
        pass


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
def classify_endpoint(payload: ClassifyRequest) -> ClassifyResponse:
    result = classify(payload.sentence, early_stop=True)
    return ClassifyResponse(
        sentence=payload.sentence,
        predicted_emotion=result["predicted_emotion"],
        done_reason=result.get("done_reason"),
    )


@app.post("/classify/stream")
def classify_stream(payload: ClassifyRequest) -> StreamingResponse:
    """
    SSE stream để frontend hiển thị sớm token / kết quả.
    """
    return StreamingResponse(stream_events(payload.sentence), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
