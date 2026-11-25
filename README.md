# DTI-Net

DTI-Net is a Streamlit demonstration portal for exploring drug-target interaction concepts, molecular structures, and disease-target data.

> **Research/demo use only.** The app's prediction wrapper uses deterministic simulated values, not a trained or clinically validated model. Do not use its outputs for medical, prescribing, or other real-world treatment decisions.

## Run locally

Use Python 3.12, matching the Streamlit Community Cloud deployment runtime.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

Run the test suite with:

```powershell
python -m pytest -q
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository's `main` branch.
3. Set the entrypoint to `app.py` and select Python 3.12 in the advanced settings.
4. No API credentials are required. The disease screening view makes requests to public UniProt, ChEMBL, and Open Targets endpoints.

Keep credentials out of Git. Local `.env` files and `.streamlit/secrets.toml` are ignored; configure any future deployment secrets in the hosting platform's secret manager instead.

The bundled `disease_cache.json` provides offline demo diseases, target sequences, and compound lists. Custom disease/target lookups use Open Targets, UniProt, and batched ChEMBL requests; if ChEMBL is unavailable, bundled compounds are used when available. The disease-screening UI reports which compound source it used.
