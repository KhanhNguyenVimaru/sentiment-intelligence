# EmoGuest App

Vue 3 + Vite single-page app that talks to Gemini for emotion classification. Follow these steps to get a local build running.

## 1. Install dependencies
Make sure Node.js 20+ is available, then install once:
```bash
npm install
```

## 2. Start the app and enter your Gemini key
```bash
npm run dev
```
- Visit <http://localhost:5173> (or the port reported by Vite) and use the built-in API key field to paste your Gemini credential.
- The key is kept in the app's UI state (e.g., session/local storage); no `.env` file or environment variables are needed for the frontend anymore.
- Refresh or restart the dev server without worrying about env vars—the UI retains the value until you clear it manually.

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
- **Key not persisting in the UI**: reopen the key panel, re-enter the Gemini credential, and refresh so the client stores it again.
- **403/401 errors**: verify the Gemini key has API access and hasn't exceeded quota.
- **Type errors around `import.meta.env`**: run `npm install` to ensure `env.d.ts` is picked up by Vite/TS.

## 6. Reference
- <https://www.studocu.vn/vn/document/dai-hoc-hang-hai-viet-nam/ky-thuat-hoc-sau-va-ung-dung/bao-cao-bai-tap-lon-ky-thuat-hoc-sau-xac-dinh-cam-xuc-dl101/150377449>
