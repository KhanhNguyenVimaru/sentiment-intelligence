"""
Evaluate local Ollama model on the Hugging Face `emotion` dataset and output JSON.

Usage:
  python evaluate_emotion.py             # default 50 samples
  python evaluate_emotion.py --limit 200 # more samples

Output: JSON with each sentence and its predicted/gold emotion plus overall accuracy.
Assumes `ollama serve` is running and model `gpt-oss:20b` is available.
"""

import argparse
import json
import re
import sys
from typing import Dict, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm


API_URL = "http://localhost:11434/api/generate"
MAIN_MODEL = "gpt-oss:20b"
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def normalize_label(raw: str) -> str:
    """Clean the model response down to a known label."""
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
    """
    Parse a JSON object like {"label": "joy"} from the model response.
    Accepts plain JSON or JSON inside a Markdown code block.
    """
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return ""
    try:
        parsed = json.loads(match.group(0))
        return str(parsed.get("label", ""))
    except Exception:
        return ""


def classify(text: str, timeout: int = 120) -> Tuple[str, Dict]:
    prompt = (
        "You are an emotion classifier. Read the English sentence and respond with JSON only.\n"
        "Allowed labels: sadness, joy, love, anger, fear, surprise.\n"
        'Output format (no extra text): {"label": "<one_of_allowed_labels>"}\n'
        f"Sentence: {text}"
    )

    payload = {
        "model": MAIN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            # Reasoning model needs enough tokens to finish after 'thinking'.
            "num_predict": 128,
        },
    }
    response = requests.post(API_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    raw = data.get("response", "")
    parsed = extract_label_from_json(raw)
    return normalize_label(parsed or raw), data


def evaluate(sample_count: int) -> None:
    split = f"test[:{sample_count}]"
    dataset = load_dataset("emotion", split=split)

    correct = 0
    results = []
    for row in tqdm(dataset, desc="Evaluating"):
        gold = LABELS[row["label"]]
        predicted, raw_data = classify(row["text"])
        if predicted == "":
            print(
                f"[warn] empty prediction; done_reason={raw_data.get('done_reason')} raw_response={raw_data}",
                file=sys.stderr,
            )
        if predicted == gold:
            correct += 1
        results.append(
            {
                "sentence": row["text"],
                "predicted_emotion": predicted,
                "gold_emotion": gold,
            }
        )

    accuracy = 100 * correct / len(dataset)
    summary = {
        "accuracy": round(accuracy, 2),
        "correct": correct,
        "total": len(dataset),
        "split": split,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gpt-oss:20b on the emotion dataset.")
    parser.add_argument("--limit", type=int, default=50, help="Number of test samples to evaluate.")
    args = parser.parse_args()
    evaluate(args.limit)


if __name__ == "__main__":
    main()
