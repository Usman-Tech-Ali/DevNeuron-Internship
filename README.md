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
- I am unable to deploy

## FGSM (brief)
FGSM perturbs input x in the direction of the gradient sign of the loss w.r.t. x: x_adv = x + ε * sign(∇_x L(θ, x, y)). It creates minimal changes that can flip model predictions.

## Observations
- Increase ε → stronger perturbations → higher misclassification likelihood.
- Summarize accuracy drop and example predictions here.



## Screenshots to Include
<img width="1078" height="892" alt="image" src="https://github.com/user-attachments/assets/0337d608-ac1d-4236-8be8-a29b41136d06" />
<img width="961" height="285" alt="image" src="https://github.com/user-attachments/assets/ef0a0f83-cc5f-417d-9194-44bc3824f007" />


## Attribution
Provide links to any external references used.
