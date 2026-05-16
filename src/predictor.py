
import numpy as np
import pandas as pd
from scipy.stats import poisson
import json

def load_model(params_path, features_path):
    with open(params_path, "r") as f:
        model_data = json.load(f)
    team_features = pd.read_csv(features_path, index_col="team")
    return model_data, team_features

def predict_match(home_team, away_team, model_data,
                  team_features=None, neutral=False,
                  elo_weight=0.3):
    attack   = model_data["attack"].copy()
    defence  = model_data["defence"].copy()
    home_adv = model_data["home_advantage"]
    rho      = model_data["rho"]

    avg_attack  = np.mean(list(attack.values()))
    avg_defence = np.mean(list(defence.values()))
    for team in [home_team, away_team]:
        if team not in attack:
            attack[team]  = avg_attack
            defence[team] = avg_defence

    home_adv_factor = 0 if neutral else home_adv
    mu_h = np.exp(attack[home_team]  - defence[away_team] + home_adv_factor)
    mu_a = np.exp(attack[away_team]  - defence[home_team])

    if team_features is not None:
        if home_team in team_features.index and away_team in team_features.index:
            elo_h  = team_features.loc[home_team, "elo"]
            elo_a  = team_features.loc[away_team, "elo"]
            form_h = team_features.loc[home_team, "form"]
            form_a = team_features.loc[away_team, "form"]

            elo_diff = (elo_h - elo_a) / 400
            elo_adj  = 10 ** (elo_diff * elo_weight)
            form_adj = ((form_h + 0.5) / (form_a + 0.5)) ** 0.2

            mu_h *= elo_adj * form_adj
            mu_a /= elo_adj * form_adj

    max_goals    = 8
    score_matrix = np.zeros((max_goals+1, max_goals+1))

    for hg in range(max_goals+1):
        for ag in range(max_goals+1):
            p = poisson.pmf(hg, mu_h) * poisson.pmf(ag, mu_a)
            if   hg==0 and ag==0: tau = 1 - mu_h*mu_a*rho
            elif hg==0 and ag==1: tau = 1 + mu_h*rho
            elif hg==1 and ag==0: tau = 1 + mu_a*rho
            elif hg==1 and ag==1: tau = 1 - rho
            else:                 tau = 1.0
            score_matrix[hg, ag] = p * max(tau, 0)

    score_matrix /= score_matrix.sum()

    return {
        "score_matrix": score_matrix,
        "home_win": float(np.sum(np.tril(score_matrix, -1))),
        "draw":     float(np.sum(np.diag(score_matrix))),
        "away_win": float(np.sum(np.triu(score_matrix, 1))),
        "mu_h":     round(float(mu_h), 3),
        "mu_a":     round(float(mu_a), 3)
    }

def simulate_match_result(home_team, away_team, model_data,
                           team_features=None, neutral=False):
    """
    Randomly samples ONE match result from the scoreline distribution.
    Used by the Monte Carlo engine.
    Returns: (home_goals, away_goals)
    """
    pred = predict_match(home_team, away_team, model_data,
                         team_features, neutral)
    matrix = pred["score_matrix"]
    
    # Flatten and sample
    probs = matrix.flatten()
    probs /= probs.sum()
    idx   = np.random.choice(len(probs), p=probs)
    hg, ag = divmod(idx, matrix.shape[1])
    return int(hg), int(ag)
