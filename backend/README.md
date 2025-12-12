# Emotion Classifier Evaluation (Ollama)

Quick script to measure accuracy of a local Ollama model on the Hugging Face `emotion` dataset (text only).

## Prerequisites
- Python 3.10+ with `pip install -r requirements.txt`
- Ollama running locally (`ollama serve` already active) and the model pulled:
  - `ollama pull gpt-oss:20b`

## Run
```bash
cd backend
python evaluate_emotion.py --limit 50        # default 50 samples
python evaluate_emotion.py --limit 10        # quicker smoke test
```

Output is JSON containing overall accuracy and per-sample fields:
- `sentence`
- `predicted_emotion`
- `gold_emotion`

## Notes
- The model is prompted to return JSON only and uses `temperature=0` with `num_predict=128` to allow enough tokens after any “thinking” phase.
- If the model fails to emit a label, the script logs a warning to stderr showing `done_reason` and the raw Ollama response (helps detect `length` truncation).
- Dataset downloads automatically on first run (Hugging Face `emotion`).
