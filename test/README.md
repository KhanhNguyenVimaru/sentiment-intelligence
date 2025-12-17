# Test Harness

Scripts in this directory evaluate the Google Gemini API against the Hugging Face `emotion` dataset. Each request handles exactly one *block* (10 sentences) so you can throttle how much traffic you send to Gemini.

## 1. Environment setup
- Python 3.10+
- Install dependencies once:
  ```bash
  cd test
  python -m venv .venv
  .\.venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
  pip install -r requirements.txt
  ```
- Set your Gemini API key in the shell (`GEMINI_API_KEY` or `VITE_GEMINI_API_KEY`). Example on PowerShell:
  ```powershell
  $env:GEMINI_API_KEY = "AIza...."
  ```
  The scripts never print the key, so make sure your `.env` files stay untracked.

## 2. Block-based evaluation
`test_gemini_blocks.py` loads `BLOCK_SIZE = 10` samples at a time, tells Gemini to respond with JSON, and compares the predictions to the dataset.

```
python test_gemini_blocks.py --blocks 3            # process 30 samples
python test_gemini_blocks.py --blocks 5 --model gemini-1.5-flash
python test_gemini_blocks.py --blocks 2 --split validation
```

Key options:
- `--blocks`: number of 10-sample blocks to send (default 1).
- `--model`: Gemini model name (default `gemini-2.5-flash`).
- `--split`: which portion of the dataset to read (`test`, `train`, etc.).
- `--api-key`: supply a key explicitly instead of relying on env vars.

Each response must follow this schema so we can diff against the dataset:
```json
{
  "block": 1,
  "results": [
    {
      "local_id": 3,
      "dataset_index": 42,
      "sentence": "example text",
      "predicted_emotion": "joy"
    }
  ]
}
```
The script normalizes synonyms (e.g., `happy` ⇒ `joy`), tracks accuracy, and prints a JSON summary with per-sample diagnostics.

## 3. Tips
- The first run downloads the `emotion` dataset automatically.
- Gemini occasionally returns Markdown code fences; the parser tolerates them, but malformed JSON causes the run to fail early so you can adjust the prompt.
- Use smaller `--blocks` when testing prompt tweaks to avoid burning quota.
