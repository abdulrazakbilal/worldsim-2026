# 🏆 WorldSim 2026
### Live Bayesian Tournament Engine — FIFA World Cup 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://worldsim-2026.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> *10,000 Monte Carlo simulations. 48 teams. 1 champion.*

WorldSim 2026 is a live probabilistic tournament engine for the FIFA World Cup 2026. It combines a Dixon-Coles Poisson goal model with Elo rating fusion and rolling form indices to simulate the entire tournament 10,000 times — updating probabilities as real match results come in.

---

## 🔴 Live Dashboard
**[worldsim-2026.streamlit.app](https://worldsim-2026.streamlit.app)**

| Page | Description |
|------|-------------|
| 🏠 Overview | Championship probabilities for all 48 teams |
| ⚔️ Match Predictor | Head-to-head win probability for any fixture |
| 🗂️ Group Explorer | Group-by-group qualification probabilities |
| 📈 Path Probabilities | Team journey from group stage to final |

---

## 📊 Current Model Predictions
*(Pre-tournament, 10,000 simulations)*

| Rank | Team | Championship % |
|------|------|---------------|
| 🥇 | Spain | 22.3% |
| 🥈 | Argentina | 17.0% |
| 🥉 | France | 10.7% |
| 4 | Morocco | 4.8% |
| 5 | England | 4.7% |
| 6 | Netherlands | 4.2% |
| 7 | Germany | 4.2% |
| 8 | Brazil | 3.9% |

---

## 🏗️ Architecture

```
worldsim-2026/
├── data/
│   ├── raw/                    ← 49,287 international match results
│   └── processed/              ← cleaned features, model outputs
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_strength_model.ipynb
│   └── 03_monte_carlo.ipynb
├── src/
│   └── predictor.py            ← match prediction engine
├── dashboard/
│   ├── app.py                  ← Streamlit dashboard
│   └── requirements.txt
└── requirements.txt
```

---

## 🧠 Methodology

### Layer 1 — Data Pipeline
- **49,287 international matches** (1990–2026) from Kaggle
- Filtered to competitive matches: World Cup, Euros, Copa América, qualifiers
- Feature engineering: goals, results, tournament weights

### Layer 2 — Team Strength Model
**Elo Rating System**
- Custom Elo engine built from scratch across 32,101 matches
- K=32, starting rating 1500 for all teams
- Captures long-run team strength with historical momentum

**Rolling Form Index**
- Points-based form over last 10 matches (3=win, 1=draw, 0=loss)
- Captures recent momentum — critical for tournament prediction

**Dixon-Coles Poisson Goal Model**
- Fits attack/defence parameters for 211 teams
- Models P(score | team A vs team B) as independent Poisson processes
- Low-score correction via rho parameter (ρ = -0.059)
- Optimised via L-BFGS-B + Powell two-stage convergence

**Elo + Form Fusion**
- Blends Dixon-Coles historical params with live Elo + form
- Separate weights for group stage (more variance) vs knockout (Elo dominant)

### Layer 3 — Monte Carlo Simulation Engine
- **10,000 full tournament simulations** per run
- Pre-computed match probability cache (2,256 matchups) for speed
- Full 48-team format: 12 groups, best-third-place selection (FIFA rules)
- Knockout bracket: R32 → R16 → QF → SF → Final
- Extra time / penalties modelled via win probability ratio

### Layer 4 — Live Streamlit Dashboard
- 4-page navigation: Overview, Match Predictor, Group Explorer, Path Probabilities
- Dark editorial theme with teal accent
- Auto-updates when simulation results are pushed to GitHub

---

## 🚀 Run Locally

```bash
git clone https://github.com/abdulrazakbilal/worldsim-2026.git
cd worldsim-2026
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 📦 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Modelling | SciPy, NumPy, Pandas |
| Optimisation | L-BFGS-B, Powell |
| Simulation | NumPy Monte Carlo |
| Dashboard | Streamlit |
| Data | Kaggle (49k matches) |

---

## 📅 Roadmap

- [x] Data pipeline & EDA
- [x] Elo rating engine
- [x] Dixon-Coles goal model
- [x] Monte Carlo simulation engine
- [x] Streamlit dashboard
- [x] Deployment to Streamlit Cloud
- [ ] Live match result updates (June 11 onwards)
- [ ] Upset detector — model vs bookmaker odds divergence
- [ ] SHAP explainability panel
- [ ] Post-tournament calibration analysis

---

## 👤 Author

**Abdul Razak Bilal**
B.Tech CSE (AI & ML) — G. Pulla Reddy Engineering College, Kurnool

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/abdul-razak-bilal)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/abdulrazakbilal)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-teal)](https://abdulrazakbilal.github.io)

---

*Tournament starts June 11, 2026. Model updates after every match.*