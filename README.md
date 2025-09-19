# DevNeuron Assessment: FGSM Demo (FastAPI + Next.js)

## Run Locally (Windows)

1) Backend
- Install Python 3.11
- Open PowerShell in `backend/`
- Create venv: `python -m venv .venv` then `./.venv/Scripts/Activate.ps1`
- Install deps: `pip install -r requirements.txt`
- Run API: `python -m uvicorn app_fgsm:app --reload --host 0.0.0.0 --port 8000`

2) Frontend
- Install Node.js 18+
- Open PowerShell in `frontend/`
- Create `.env.local` with: `NEXT_PUBLIC_API_BASE=http://localhost:8000`
- Install deps: `npm install`
- Run dev: `npm run dev` (app at http://localhost:3000)

## Deployed URLs
- Frontend (Amplify): <add after deploy>
- Backend (API Gateway/EC2): <add after deploy>

## FGSM (brief)
FGSM perturbs input x in the direction of the gradient sign of the loss w.r.t. x: x_adv = x + ε * sign(∇_x L(θ, x, y)). It creates minimal changes that can flip model predictions.

## Observations
- Increase ε → stronger perturbations → higher misclassification likelihood.
- Summarize accuracy drop and example predictions here.

## AWS Deploy (summary)

Option A: Lambda + API Gateway (serverless)
1) Ensure `backend/lambda_handler.py` exists and `mangum` in `requirements.txt`.
2) Build Linux-compatible deps (recommend AWS Cloud9):
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r backend/requirements.txt`
   - Zip site-packages + `backend/*.py`.
3) Create Lambda (Python 3.11): upload zip, Handler `lambda_handler.handler`, Memory 2048MB, Timeout 600s.
4) Create API Gateway HTTP API → integrate Lambda. Enable CORS for your Amplify domain.
5) Use the API invoke URL as `NEXT_PUBLIC_API_BASE` in Amplify.

Option B: EC2 (always-on)
1) Launch t2.micro Amazon Linux, open ports 80/8000.
2) SSH, install Python and git, clone repo, create venv, `pip install -r backend/requirements.txt uvicorn`.
3) Run: `uvicorn app_fgsm:app --host 0.0.0.0 --port 8000` or set up systemd/Nginx.
4) Use EC2 public URL as `NEXT_PUBLIC_API_BASE` in Amplify.

## Screenshots to Include
- Local backend running, frontend running, successful attack results.
- Deployed API test (API Gateway/EC2), Amplify app performing attack.
- Accuracy drop file output.

## Attribution
Provide links to any external references used.
