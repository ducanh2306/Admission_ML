# UCLA Admission Prediction — Neural Network

**CST2216 Individual Term Project — Modularizing and Deploying ML Code**

A modular, production-style neural network project (MLPClassifier) that predicts a student's chance of admission to UCLA, deployed as a Streamlit web application. https://admissionan.streamlit.app/

---

## Project Structure

```
admission_nn/
├── app.py                   # Streamlit web application (entry point)
├── requirements.txt
├── README.md
├── data/
│   └── Admission.csv        # 500 applicants × 9 columns
├── logs/
│   └── app.log              # Auto-generated runtime log
├── src/
│   ├── __init__.py
│   ├── config.py             # Centralised configuration
│   ├── logger.py             # File + console logging
│   ├── data_loader.py        # CSV loading & validation
│   ├── preprocessing.py      # Target binarisation, encoding, scaling
│   ├── train.py               # MLP training, CV, evaluation, persistence
│   └── predict.py             # Single-applicant inference
└── tests/
    └── test_pipeline.py       # 11 pytest unit tests
```

---

## Quick Start

### 1. Open the project folder

```bash
cd admission_nn
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## App Pages

| Page | What it does |
|------|-------------|
| 📊 Data Explorer | Distributions, correlation heatmap, GRE vs TOEFL scatter coloured by admission outcome |
| 🧠 Model Training | Configure hidden layers, neurons per layer, activation function, batch size, and max iterations from the sidebar; train and view loss curve, confusion matrices, classification reports, and cross-validation scores |
| 🔮 Predict | Enter a single applicant's profile (GRE, TOEFL, CGPA, SOP, LOR, research, university rating) and get an instant admit/reject prediction with probability |

---

## Dataset

UCLA Admission dataset — 500 applicants, 9 columns, no missing values.

**Target:** `Admit_Chance` — binarised at **0.80** threshold (≥0.80 → Admit = 1) per project specification.

| Column | Description |
|--------|-------------|
| GRE_Score | Out of 340 |
| TOEFL_Score | Out of 120 |
| University_Rating | Bachelor university ranking (1–5, treated as categorical) |
| SOP | Statement of Purpose strength (out of 5) |
| LOR | Letter of Recommendation strength (out of 5) |
| CGPA | Undergraduate GPA (out of 10) |
| Research | Research experience (0/1, treated as categorical) |
| Admit_Chance | Original continuous probability (0–1) |

---

## Model

**MLPClassifier** (feed-forward neural network), scikit-learn.

Default architecture (matches the solution notebook): 1 hidden layer, 3 neurons, ReLU activation, batch size 50, 200 max iterations. All of these are tunable live from the Streamlit sidebar.

**Success criteria (per project spec):** Test accuracy ≥ 90%.

---

## Dependencies

- Python ≥ 3.10
- streamlit, pandas, numpy, scikit-learn, matplotlib, seaborn, pytest


