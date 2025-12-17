# EmoGuest Frontend

Vue 3 + Vite single-page app that talks to Gemini for emotion classification. Follow these steps to get a local build running.

## 1. Create `.env` from `.env.example`
1. Copy the template:
   ```bash
   cd app
   cp .env.example .env
   ```
2. Open `app/.env` and replace the placeholder key with your real Gemini API key in both variables:
   ```env
   GEMINI_API_KEY=your_real_key_here
   VITE_GEMINI_API_KEY=your_real_key_here
   ```
   - The duplicated values keep both the frontend build step (`VITE_...`) and any Node tooling aligned.
   - Never commit `.env`; it is excluded via `.gitignore`.
3. Restart `npm run dev` whenever you touch `.env`, since Vite only loads env vars at startup.

## 2. Install dependencies
Make sure Node.js 20+ is available, then install once:
```bash
npm install
```

## 3. Useful scripts
| Command | Description |
| --- | --- |
| `npm run dev` | Hot-reload dev server (http://localhost:5173) |
| `npm run type-check` | TypeScript diagnostics via `vue-tsc` |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview the production bundle |

## 4. Test the Gemini model
Use the Python harness under `../test` when you need to validate prompt quality or measure accuracy.

1. Create/activate a virtual environment inside `test/`.
   ```bash
   cd ../test
   python -m venv .venv
   .\.venv\Scripts\activate  # or source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Export the same Gemini API key the frontend uses:
   ```powershell
   $env:GEMINI_API_KEY = "your_real_key_here"
   ```
3. Run the block-based evaluator (each block = 10 samples):
   ```bash
   python test_gemini_blocks.py --blocks 3 --model gemini-2.5-flash
   ```
   The script prints JSON containing every sentence, predicted emotion, and accuracy metrics so you can compare with the Hugging Face dataset.

## 5. Troubleshooting
- **Missing VITE_GEMINI_API_KEY**: confirm `.env` exists and the server was restarted after editing it.
- **403/401 errors**: verify the Gemini key has API access and hasn’t exceeded quota.
- **Type errors around `import.meta.env`**: run `npm install` to ensure `env.d.ts` is picked up by Vite/TS.
