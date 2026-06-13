# Conformal Prediction with Macro‑Adaptive Non‑conformity

Provides distribution‑free prediction intervals for ETF returns using conformal prediction. The non‑conformity score is weighted by a composite macro factor (from VIX, DXY, yields), adapting the interval width to market conditions. The per‑ETF score is the confidence (1 / (1 + interval width)) – higher means a narrower, more confident prediction.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Random forest regressor trained on macro variables
- Non‑conformity = |residual| / (1 + macro_factor)
- Conformal quantile threshold ensures coverage
- Score = confidence (narrow intervals = high confidence)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-conformal-prediction-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High confidence → ETF's next‑day return is highly predictable from macro.
- Low confidence → uncertain prediction.

## Requirements

See `requirements.txt`.
