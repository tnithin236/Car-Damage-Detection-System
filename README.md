# AI Car Damage Detection & Severity Analysis

YOLO-based vehicle damage detection (and segmentation when the trained model supports it), with a rule-based
severity estimate, damage-area estimate, and downloadable inspection report, served through Streamlit.

## Dataset
CarDD (Wang, Li, Wu; IEEE T-ITS, 2023), via Kaggle. Check the dataset license before redistributing.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# copy your trained weights to models/best.pt
streamlit run app.py
```
Custom weights path: `DAMAGE_MODEL_PATH=path/to/best.pt streamlit run app.py`

## Model performance
Paste the real numbers from `metrics.json` produced by the training notebook. Do not estimate them.

## Limitations
- Severity and damage area are AI estimates, not insurance or mechanical assessments.
- Area is relative to the whole image (no vehicle segmentation).
- No component names (e.g. "front bumper"): the dataset labels damage type only.
- Repair cost is not estimated (no labeled cost data).
