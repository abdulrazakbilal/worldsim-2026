import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from datetime import date, timedelta, datetime
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

st.set_page_config(
    page_title="WorldSim 2026",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 8px; padding: 10px; }
    h1 { color: #00d4a4; font-family: 'Space Mono', monospace; }
    h2 { color: #ffffff; }
    h3 { color: #00d4a4; }
</style>
""", unsafe_allow_html=True)

# ── WC Groups ────────────────────────────────────────────────
WC2026_GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Norway', 'Iraq'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}
all_wc_teams = sorted([t for grp in WC2026_GROUPS.values() for t in grp])

# ── OFFICIAL WC 2026 Fixture Schedule (verified ESPN/Yahoo) ──
WC_FIXTURES = [
    # June 11
    ("Mexico",        "South Africa",           "A", "2026-06-11"),
    ("South Korea",   "Czechia",                "A", "2026-06-11"),
    # June 12
    ("Canada",        "Bosnia and Herzegovina", "B", "2026-06-12"),
    ("Qatar",         "Switzerland",            "B", "2026-06-12"),
    ("United States", "Paraguay",               "D", "2026-06-12"),
    ("Australia",     "Turkey",                 "D", "2026-06-12"),
    # June 13
    ("Brazil",        "Morocco",                "C", "2026-06-13"),
    ("Haiti",         "Scotland",               "C", "2026-06-13"),
    # June 14
    ("Germany",       "Curacao",                "E", "2026-06-14"),
    ("Ivory Coast",   "Ecuador",                "E", "2026-06-14"),
    ("Netherlands",   "Japan",                  "F", "2026-06-14"),
    ("Sweden",        "Tunisia",                "F", "2026-06-14"),
    # June 15
    ("Spain",         "Cape Verde",             "H", "2026-06-15"),
    ("Saudi Arabia",  "Uruguay",                "H", "2026-06-15"),
    ("Belgium",       "Egypt",                  "G", "2026-06-15"),
    ("Iran",          "New Zealand",            "G", "2026-06-15"),
    # June 16
    ("France",        "Senegal",                "I", "2026-06-16"),
    ("Iraq",          "Norway",                 "I", "2026-06-16"),
    ("Argentina",     "Algeria",                "J", "2026-06-16"),
    ("Austria",       "Jordan",                 "J", "2026-06-16"),
    # June 17
    ("Portugal",      "DR Congo",               "K", "2026-06-17"),
    ("Uzbekistan",    "Colombia",               "K", "2026-06-17"),
    ("England",       "Croatia",                "L", "2026-06-17"),
    ("Ghana",         "Panama",                 "L", "2026-06-17"),
    # June 18
    ("Czechia",       "South Africa",           "A", "2026-06-18"),
    ("Mexico",        "South Korea",            "A", "2026-06-18"),
    ("Switzerland",   "Bosnia and Herzegovina", "B", "2026-06-18"),
    ("Canada",        "Qatar",                  "B", "2026-06-18"),
    # June 19
    ("Paraguay",      "Australia",              "D", "2026-06-19"),
    ("United States", "Turkey",                 "D", "2026-06-19"),
    ("Scotland",      "Morocco",                "C", "2026-06-19"),
    ("Brazil",        "Haiti",                  "C", "2026-06-19"),
    # June 20
    ("Ecuador",       "Germany",                "E", "2026-06-20"),
    ("Curacao",       "Ivory Coast",            "E", "2026-06-20"),
    ("Tunisia",       "Netherlands",            "F", "2026-06-20"),
    ("Japan",         "Sweden",                 "F", "2026-06-20"),
    # June 21
    ("Cape Verde",    "Saudi Arabia",           "H", "2026-06-21"),
    ("Uruguay",       "Spain",                  "H", "2026-06-21"),
    ("Egypt",         "New Zealand",            "G", "2026-06-21"),
    ("Belgium",       "Iran",                   "G", "2026-06-21"),
    # June 22
    ("Norway",        "France",                 "I", "2026-06-22"),
    ("Senegal",       "Iraq",                   "I", "2026-06-22"),
    ("Algeria",       "Austria",                "J", "2026-06-22"),
    ("Jordan",        "Argentina",              "J", "2026-06-22"),
    # June 23
    ("Colombia",      "Portugal",               "K", "2026-06-23"),
    ("DR Congo",      "Uzbekistan",             "K", "2026-06-23"),
    ("Croatia",       "Ghana",                  "L", "2026-06-23"),
    ("England",       "Panama",                 "L", "2026-06-23"),
    # June 24
    ("South Africa",  "South Korea",            "A", "2026-06-24"),
    ("Czechia",       "Mexico",                 "A", "2026-06-24"),
    ("Bosnia and Herzegovina", "Qatar",         "B", "2026-06-24"),
    ("Switzerland",   "Canada",                 "B", "2026-06-24"),
    # June 25
    ("Turkey",        "Paraguay",               "D", "2026-06-25"),
    ("Australia",     "United States",          "D", "2026-06-25"),
    ("Morocco",       "Haiti",                  "C", "2026-06-25"),
    ("Scotland",      "Brazil",                 "C", "2026-06-25"),
    # June 26
    ("Ivory Coast",   "Curacao",                "E", "2026-06-26"),
    ("Germany",       "Ecuador",                "E", "2026-06-26"),
    ("Japan",         "Tunisia",                "F", "2026-06-26"),
    ("Sweden",        "Netherlands",            "F", "2026-06-26"),
    # June 27
    ("New Zealand",   "Belgium",                "G", "2026-06-27"),
    ("Egypt",         "Iran",                   "G", "2026-06-27"),
    ("Cape Verde",    "Uruguay",                "H", "2026-06-27"),
    ("Saudi Arabia",  "Spain",                  "H", "2026-06-27"),
    ("Iraq",          "France",                 "I", "2026-06-27"),
    ("Norway",        "Senegal",                "I", "2026-06-27"),
    ("Algeria",       "Jordan",                 "J", "2026-06-27"),
    ("Austria",       "Argentina",              "J", "2026-06-27"),
    ("DR Congo",      "Colombia",               "K", "2026-06-27"),
    ("Uzbekistan",    "Portugal",               "K", "2026-06-27"),
    ("Ghana",         "Croatia",                "L", "2026-06-27"),
    ("Panama",        "England",                "L", "2026-06-27"),
]

# ── Session state for live results ───────────────────────────
if 'results_log' not in st.session_state:
    st.session_state.results_log = []  # list of (home, away, hg, ag)

if 'live_elo' not in st.session_state:
    st.session_state.live_elo = {}  # will be populated after load

if 'points_table' not in st.session_state:
    # Initialize empty points table for all teams
    st.session_state.points_table = {
        team: {'pts': 0, 'w': 0, 'd': 0, 'l': 0,
               'gf': 0, 'ga': 0, 'gd': 0, 'gp': 0}
        for grp in WC2026_GROUPS.values() for team in grp
    }

# ── Load data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    results_df    = pd.read_csv(os.path.join(base, 'simulation_results.csv'))
    team_features = pd.read_csv(os.path.join(base, 'team_features.csv'),
                                index_col='team')
    with open(os.path.join(base, 'dixon_coles_params.json'), 'r') as f:
        model_data = json.load(f)
    return results_df, team_features, model_data

@st.cache_data
def build_match_cache(_team_features):
    tf = _team_features.copy()
    if 'Curacao' not in tf.index:
        tf.loc['Curacao'] = {'elo': 1480, 'form': 1.2}
    if 'Czechia' not in tf.index:
        tf.loc['Czechia'] = {'elo': 1780, 'form': 1.8}

    def elo_win_prob(home, away, tf_local):
        if home not in tf_local.index or away not in tf_local.index:
            return 0.4, 0.2, 0.4
        elo_h    = tf_local.loc[home, 'elo']
        elo_a    = tf_local.loc[away, 'elo']
        form_h   = tf_local.loc[home, 'form']
        form_a   = tf_local.loc[away, 'form']
        exp_h    = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        form_adj = ((form_h + 0.5) / (form_a + 0.5)) ** 0.15
        exp_h    = float(np.clip(exp_h * form_adj, 0.05, 0.95))
        draw     = float(np.clip(0.28 * (1 - abs(exp_h - 0.5) * 1.2), 0.18, 0.30))
        win_h    = exp_h * (1 - draw)
        win_a    = (1 - exp_h) * (1 - draw)
        return win_h, draw, win_a

    cache_knockout = {}
    for home in all_wc_teams:
        for away in all_wc_teams:
            if home != away:
                w, d, l = elo_win_prob(home, away, tf)
                cache_knockout[(home, away)] = {
                    'home_win': w, 'draw': d, 'away_win': l,
                    'mu_h': max(w*1.5, 0.3), 'mu_a': max(l*1.5, 0.3)
                }
    return cache_knockout, tf

results_df, team_features, model_data = load_data()

with st.spinner("⚙️ Building match probability cache..."):
    match_cache_knockout, team_features = build_match_cache(team_features)

# Initialize live_elo from team_features if empty
if not st.session_state.live_elo:
    st.session_state.live_elo = team_features['elo'].to_dict()

def get_live_pred(home, away):
    """Get win probability using live (updated) Elo ratings."""
    elo_h  = st.session_state.live_elo.get(home, 1500)
    elo_a  = st.session_state.live_elo.get(away, 1500)
    exp_h  = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    exp_h  = float(np.clip(exp_h, 0.05, 0.95))
    draw   = float(np.clip(0.28 * (1 - abs(exp_h - 0.5) * 1.2), 0.18, 0.30))
    win_h  = exp_h * (1 - draw)
    win_a  = (1 - exp_h) * (1 - draw)
    return win_h, draw, win_a

def update_elo_after_result(home, away, hg, ag):
    """Update live Elo ratings after a match result."""
    elo_h = st.session_state.live_elo.get(home, 1500)
    elo_a = st.session_state.live_elo.get(away, 1500)
    exp_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
    if hg > ag:
        s_h, s_a = 1.0, 0.0
    elif ag > hg:
        s_h, s_a = 0.0, 1.0
    else:
        s_h, s_a = 0.5, 0.5
    k = 32
    st.session_state.live_elo[home] = elo_h + k * (s_h - exp_h)
    st.session_state.live_elo[away] = elo_a + k * (s_a - (1 - exp_h))

def update_points_table(home, away, hg, ag):
    """Update group standings after a result."""
    t = st.session_state.points_table
    t[home]['gp'] += 1; t[away]['gp'] += 1
    t[home]['gf'] += hg; t[home]['ga'] += ag
    t[away]['gf'] += ag; t[away]['ga'] += hg
    t[home]['gd'] = t[home]['gf'] - t[home]['ga']
    t[away]['gd'] = t[away]['gf'] - t[away]['ga']
    if hg > ag:
        t[home]['pts'] += 3; t[home]['w'] += 1; t[away]['l'] += 1
    elif ag > hg:
        t[away]['pts'] += 3; t[away]['w'] += 1; t[home]['l'] += 1
    else:
        t[home]['pts'] += 1; t[home]['d'] += 1
        t[away]['pts'] += 1; t[away]['d'] += 1

def match_already_entered(home, away):
    """Check if a match result has already been entered."""
    for r in st.session_state.results_log:
        if r[0] == home and r[1] == away:
            return True
    return False

def get_completed_teams():
    """Teams that have played at least one match."""
    completed = set()
    for r in st.session_state.results_log:
        completed.add(r[0]); completed.add(r[1])
    return completed

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏆 WorldSim 2026")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "⚔️ Match Predictor",
    "🗂️ Group Explorer",
    "📈 Path Probabilities",
    "🔄 Update Results",
    "📊 Live Standings",
    "📅 Today's Matches",
    "📖 Methodology",
    "🏟️ Bracket",
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Elo + Form fusion")
st.sidebar.markdown("- Dixon-Coles goal model")
st.sidebar.markdown("- 10,000 Monte Carlo sims")
st.sidebar.markdown(f"- Matches entered: {len(st.session_state.results_log)}")

# ── Header ────────────────────────────────────────────────────
st.title("🏆 WorldSim 2026")
st.markdown("### Live Bayesian Tournament Engine — FIFA World Cup 2026")
st.markdown("---")

top3 = results_df.head(3)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🥇 Favourite",     top3.iloc[0]['team'], f"{top3.iloc[0]['champion%']:.1f}%")
with col2:
    st.metric("🥈 2nd Favourite", top3.iloc[1]['team'], f"{top3.iloc[1]['champion%']:.1f}%")
with col3:
    st.metric("🥉 3rd Favourite", top3.iloc[2]['team'], f"{top3.iloc[2]['champion%']:.1f}%")
with col4:
    st.metric("🎲 Simulations", "10,000", "Monte Carlo")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.subheader("📊 Championship Probabilities")
    top15 = results_df.head(15)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e2130')
    colors = ['#FFD700' if i==0 else '#C0C0C0' if i==1
              else '#CD7F32' if i==2 else '#00d4a4'
              for i in range(len(top15))]
    bars = ax.barh(top15['team'][::-1], top15['champion%'][::-1],
                   color=colors[::-1], edgecolor='none')
    for bar, val in zip(bars, top15['champion%'][::-1]):
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white', fontsize=10)
    ax.set_xlabel('Championship Probability (%)', color='white')
    ax.set_title('WorldSim 2026 — Championship Probabilities\n(10,000 Monte Carlo Simulations)',
                 color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.grid(axis='x', alpha=0.2, color='white')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("---")
    st.subheader("📋 Full Tournament Path Probabilities")
    st.dataframe(
        results_df.style.format({c: '{:.1f}%' for c in results_df.columns if '%' in c})
        .background_gradient(subset=['champion%'], cmap='YlOrRd'),
        use_container_width=True, height=500
    )

# ══════════════════════════════════════════════════════════════
# PAGE: MATCH PREDICTOR
# ══════════════════════════════════════════════════════════════
elif page == "⚔️ Match Predictor":
    st.subheader("⚔️ Head-to-Head Match Predictor")
    st.markdown("Probabilities update in real-time based on results entered in 🔄 Update Results.")

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Team A", all_wc_teams,
                                  index=all_wc_teams.index('Argentina'))
    with col2:
        away_team = st.selectbox("✈️ Team B", all_wc_teams,
                                  index=all_wc_teams.index('France'))

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        hw, dr, aw = get_live_pred(home_team, away_team)
        hw *= 100; dr *= 100; aw *= 100

        st.markdown(f"### {home_team}  vs  {away_team}")

        # Show if Elo has been updated
        if len(st.session_state.results_log) > 0:
            st.info(f"📡 Using live Elo ratings — updated after "
                   f"{len(st.session_state.results_log)} match(es)")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric(f"🟢 {home_team}", f"{hw:.1f}%")
        with c2: st.metric("🤝 Draw",          f"{dr:.1f}%")
        with c3: st.metric(f"🔴 {away_team}",  f"{aw:.1f}%")

        fig2, ax2 = plt.subplots(figsize=(10, 1.5))
        fig2.patch.set_facecolor('#0e1117')
        ax2.set_facecolor('#0e1117')
        ax2.barh([0], [hw],    color='#00d4a4')
        ax2.barh([0], [dr],    left=[hw],    color='#888888')
        ax2.barh([0], [aw],    left=[hw+dr], color='#FF6B6B')
        ax2.set_xlim(0, 100); ax2.set_yticks([])
        ax2.spines[:].set_visible(False)
        ax2.tick_params(colors='white')
        for val, left, clr in [(hw, hw/2, 'black'),
                               (dr, hw+dr/2, 'white'),
                               (aw, hw+dr+aw/2, 'black')]:
            if val > 8:
                ax2.text(left, 0, f'{val:.0f}%', ha='center',
                        va='center', color=clr, fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("#### Tournament path comparison")
        h_row = results_df[results_df['team']==home_team]
        a_row = results_df[results_df['team']==away_team]
        if not h_row.empty and not a_row.empty:
            h_row = h_row.iloc[0]; a_row = a_row.iloc[0]
            stages = ['reach_R16%','reach_QF%','reach_SF%','reach_Final%','champion%']
            labels = ['R16','QF','SF','Final','🏆']
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            fig3.patch.set_facecolor('#0e1117'); ax3.set_facecolor('#1e2130')
            x = np.arange(len(labels)); w = 0.35
            ax3.bar(x-w/2, [h_row[s] for s in stages], w,
                   label=home_team, color='#00d4a4')
            ax3.bar(x+w/2, [a_row[s] for s in stages], w,
                   label=away_team, color='#FF6B6B')
            ax3.set_xticks(x); ax3.set_xticklabels(labels, color='white', fontsize=12)
            ax3.set_ylabel('Probability (%)', color='white')
            ax3.set_title('Tournament Path Comparison', color='white', fontweight='bold')
            ax3.legend(facecolor='#1e2130', labelcolor='white')
            ax3.tick_params(colors='white'); ax3.spines[:].set_color('#444')
            ax3.grid(axis='y', alpha=0.2, color='white')
            plt.tight_layout(); st.pyplot(fig3)

# ══════════════════════════════════════════════════════════════
# PAGE: GROUP EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "🗂️ Group Explorer":
    st.subheader("🗂️ Group Stage Explorer")
    selected_group = st.selectbox("Select Group",
        [f"Group {k} — {' | '.join(v)}" for k, v in WC2026_GROUPS.items()])
    group_letter = selected_group.split()[1]
    group_teams  = WC2026_GROUPS[group_letter]
    group_data   = results_df[results_df['team'].isin(group_teams)]\
                   .sort_values('group_qualify%', ascending=False)

    cols = st.columns(4)
    for i, (_, row) in enumerate(group_data.iterrows()):
        with cols[i]:
            st.metric(row['team'],
                      f"{row['group_qualify%']:.0f}% qualify",
                      f"Champion: {row['champion%']:.1f}%")
    st.markdown("---")

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.patch.set_facecolor('#0e1117'); ax4.set_facecolor('#1e2130')
    teams = group_data['team'].tolist(); x = np.arange(len(teams)); w = 0.25
    ax4.bar(x-w, group_data['group_qualify%'].values, w, label='Qualify', color='#00d4a4')
    ax4.bar(x,   group_data['reach_QF%'].values,      w, label='Reach QF', color='#FFD700')
    ax4.bar(x+w, group_data['champion%'].values,      w, label='Champion', color='#FF6B6B')
    ax4.set_xticks(x); ax4.set_xticklabels(teams, color='white', fontsize=11)
    ax4.set_ylabel('Probability (%)', color='white')
    ax4.set_title(f'Group {group_letter} — Team Probabilities',
                 color='white', fontweight='bold', fontsize=13)
    ax4.legend(facecolor='#1e2130', labelcolor='white')
    ax4.tick_params(colors='white'); ax4.spines[:].set_color('#444')
    ax4.grid(axis='y', alpha=0.2, color='white')
    plt.tight_layout(); st.pyplot(fig4)

    st.markdown("---")
    st.dataframe(
        group_data[['team','group_qualify%','reach_R16%','reach_QF%',
                    'reach_SF%','reach_Final%','champion%']]
        .style.format({c: '{:.1f}%' for c in group_data.columns if '%' in c})
        .background_gradient(subset=['group_qualify%'], cmap='Greens'),
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════
# PAGE: PATH PROBABILITIES
# ══════════════════════════════════════════════════════════════
elif page == "📈 Path Probabilities":
    st.subheader("📈 Team Tournament Path")
    selected_team = st.selectbox("Select Team", results_df['team'].tolist())
    row    = results_df[results_df['team']==selected_team].iloc[0]
    stages = ['group_qualify%','reach_R16%','reach_QF%',
              'reach_SF%','reach_Final%','champion%']
    labels = ['Qualify','R16','QF','SF','Final','🏆 Win']
    values = [row[s] for s in stages]

    cols = st.columns(6)
    for i, (lbl, val) in enumerate(zip(labels, values)):
        with cols[i]: st.metric(lbl, f"{val:.1f}%")

    # Show live Elo
    if selected_team in st.session_state.live_elo:
        orig_elo = team_features.loc[selected_team, 'elo'] if selected_team in team_features.index else 1500
        live_elo = st.session_state.live_elo[selected_team]
        delta    = live_elo - orig_elo
        if abs(delta) > 0.1:
            st.info(f"📡 {selected_team} Elo: {orig_elo:.0f} → **{live_elo:.0f}** ({delta:+.1f} after tournament results)")

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    fig5.patch.set_facecolor('#0e1117'); ax5.set_facecolor('#1e2130')
    ax5.plot(labels, values, color='#00d4a4', linewidth=2.5, marker='o', markersize=8)
    ax5.fill_between(range(len(labels)), values, alpha=0.15, color='#00d4a4')
    for i, val in enumerate(values):
        ax5.annotate(f'{val:.1f}%', (i, val), textcoords="offset points",
                    xytext=(0, 12), ha='center', color='white', fontsize=10)
    ax5.set_xticks(range(len(labels))); ax5.set_xticklabels(labels, color='white', fontsize=11)
    ax5.set_ylabel('Probability (%)', color='white')
    ax5.set_title(f'{selected_team} — Tournament Path Probabilities',
                 color='white', fontweight='bold', fontsize=13)
    ax5.tick_params(colors='white'); ax5.spines[:].set_color('#444')
    ax5.grid(alpha=0.2, color='white')
    plt.tight_layout(); st.pyplot(fig5)

# ══════════════════════════════════════════════════════════════
# PAGE: UPDATE RESULTS
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Update Results":
    st.subheader("🔄 Live Match Result Entry")
    st.markdown("Enter real match results. Elo ratings update automatically after each result.")

    if st.session_state.results_log:
        st.success(f"✅ {len(st.session_state.results_log)} result(s) entered this session")
        with st.expander("View entered results"):
            for r in st.session_state.results_log:
                st.markdown(f"- **{r[0]} {r[2]} – {r[3]} {r[1]}**")

    st.markdown("---")
    st.markdown("#### Select a fixture to update")

    # Show only fixtures that haven't been entered yet
    today_str = date.today().strftime("%Y-%m-%d")
    available = [(ta, tb, grp, d) for ta, tb, grp, d in WC_FIXTURES
                 if d <= today_str and not match_already_entered(ta, tb)]
    completed_fixtures = [(ta, tb, grp, d) for ta, tb, grp, d in WC_FIXTURES
                          if match_already_entered(ta, tb)]

    if available:
        fixture_labels = [f"Group {grp} | {ta} vs {tb} ({d})"
                         for ta, tb, grp, d in available]
        selected_fixture = st.selectbox("Choose match", fixture_labels)
        idx = fixture_labels.index(selected_fixture)
        sel_home, sel_away, sel_grp, sel_date = available[idx]

        # Show pre-match probability
        hw, dr, aw = get_live_pred(sel_home, sel_away)
        st.markdown(f"**Pre-match odds:** {sel_home} {hw*100:.0f}% | "
                   f"Draw {dr*100:.0f}% | {sel_away} {aw*100:.0f}%")

        col4, col5 = st.columns(2)
        with col4:
            score_a = st.number_input(f"{sel_home} goals",
                                       min_value=0, max_value=20, value=0, key="score_h")
        with col5:
            score_b = st.number_input(f"{sel_away} goals",
                                       min_value=0, max_value=20, value=0, key="score_a")

        if st.button("⚡ Submit Result", type="primary"):
            hg, ag = int(score_a), int(score_b)

            # Update Elo
            old_elo_h = st.session_state.live_elo.get(sel_home, 1500)
            old_elo_a = st.session_state.live_elo.get(sel_away, 1500)
            update_elo_after_result(sel_home, sel_away, hg, ag)
            new_elo_h = st.session_state.live_elo[sel_home]
            new_elo_a = st.session_state.live_elo[sel_away]

            # Update points table
            update_points_table(sel_home, sel_away, hg, ag)

            # Log result
            st.session_state.results_log.append((sel_home, sel_away, hg, ag))

            # Show result
            if hg > ag:
                result_str = f"✅ {sel_home} WIN"
            elif ag > hg:
                result_str = f"✅ {sel_away} WIN"
            else:
                result_str = "🤝 DRAW"

            st.markdown(f"### {sel_home} {hg} — {ag} {sel_away}")
            st.markdown(f"**{result_str}**")
            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                delta_h = new_elo_h - old_elo_h
                st.metric(sel_home, f"{new_elo_h:.0f}", f"{delta_h:+.1f}")
                st.markdown(f"*{old_elo_h:.0f} → {new_elo_h:.0f}*")
            with col2:
                delta_a = new_elo_a - old_elo_a
                st.metric(sel_away, f"{new_elo_a:.0f}", f"{delta_a:+.1f}")
                st.markdown(f"*{old_elo_a:.0f} → {new_elo_a:.0f}*")

            # Upset analysis
            if hg > ag:
                actual_winner = sel_home
            elif ag > hg:
                actual_winner = sel_away
            else:
                actual_winner = "Draw"

            model_fav = sel_home if hw > aw else sel_away

            st.markdown("---")
            st.markdown("#### 🚨 Upset Analysis")
            if actual_winner != "Draw" and actual_winner != model_fav:
                upset_prob = aw if actual_winner == sel_away else hw
                st.error(f"⚠️ UPSET! Model gave {actual_winner} only "
                        f"{upset_prob*100:.1f}% chance of winning.")
            elif actual_winner == "Draw":
                st.info(f"Draw — model gave {dr*100:.1f}% probability.")
            else:
                fav_prob = hw if model_fav == sel_home else aw
                st.success(f"✅ Favourite won ({fav_prob*100:.1f}% pre-match).")

            st.info("💡 Go to **⚔️ Match Predictor** to see updated probabilities, "
                   "or **📊 Live Standings** for the current group table.")
            st.rerun()
    else:
        if not WC_FIXTURES:
            st.info("No fixtures available yet.")
        else:
            st.success("✅ All available fixtures entered! Check back after more matches.")

    if completed_fixtures:
        st.markdown("---")
        st.markdown("#### ✅ Already Entered")
        for ta, tb, grp, d in completed_fixtures:
            for r in st.session_state.results_log:
                if r[0] == ta and r[1] == tb:
                    st.markdown(f"Group {grp} | {ta} **{r[2]}–{r[3]}** {tb} ✓")

    if st.button("🔄 Reset All Results", type="secondary"):
        st.session_state.results_log = []
        st.session_state.live_elo = team_features['elo'].to_dict()
        st.session_state.points_table = {
            team: {'pts': 0, 'w': 0, 'd': 0, 'l': 0,
                   'gf': 0, 'ga': 0, 'gd': 0, 'gp': 0}
            for grp in WC2026_GROUPS.values() for team in grp
        }
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: LIVE STANDINGS
# ══════════════════════════════════════════════════════════════
elif page == "📊 Live Standings":
    st.subheader("📊 Live Group Stage Standings")

    if not st.session_state.results_log:
        st.info("No results entered yet. Go to **🔄 Update Results** to enter match results.")
    else:
        st.markdown(f"*Updated after {len(st.session_state.results_log)} match(es)*")

    for grp_letter, teams in WC2026_GROUPS.items():
        st.markdown(f"### Group {grp_letter}")
        table_data = []
        for team in teams:
            t = st.session_state.points_table[team]
            table_data.append({
                'Team': team,
                'GP': t['gp'], 'W': t['w'], 'D': t['d'], 'L': t['l'],
                'GF': t['gf'], 'GA': t['ga'], 'GD': t['gd'], 'Pts': t['pts']
            })
        table_df = pd.DataFrame(table_data)\
                   .sort_values(['Pts','GD','GF'], ascending=False)\
                   .reset_index(drop=True)
        table_df.index = table_df.index + 1

        # Highlight top 2
        def highlight_top2(row):
            if row.name <= 2:
                return ['background-color: #1a3a2a'] * len(row)
            return [''] * len(row)

        st.dataframe(
            table_df.style.apply(highlight_top2, axis=1),
            use_container_width=True,
            hide_index=False
        )
        st.markdown("")

# ══════════════════════════════════════════════════════════════
# PAGE: TODAY'S MATCHES
# ══════════════════════════════════════════════════════════════
elif page == "📅 Today's Matches":
    st.subheader("📅 Today's World Cup Fixtures")

    today        = date.today()
    today_str    = today.strftime("%Y-%m-%d")
    tomorrow_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

    todays    = [(ta,tb,grp,d) for ta,tb,grp,d in WC_FIXTURES if d == today_str]
    tomorrows = [(ta,tb,grp,d) for ta,tb,grp,d in WC_FIXTURES if d == tomorrow_str]

    if todays:
        st.markdown(f"### 🔴 Today — {today.strftime('%B %d, %Y')}")
        for ta, tb, grp, d in todays:
            hw, dr, aw = get_live_pred(ta, tb)
            hw *= 100; dr *= 100; aw *= 100
            done = match_already_entered(ta, tb)
            st.markdown(f"**Group {grp}** {'✅ Result entered' if done else '⏳ Upcoming'}")
            c1,c2,c3,c4,c5 = st.columns([3,1,3,1,3])
            with c1: st.metric(ta, f"{hw:.0f}%", "Win")
            with c2: st.markdown("<br><br>**vs**", unsafe_allow_html=True)
            with c3: st.metric(tb, f"{aw:.0f}%", "Win")
            with c4: st.markdown("<br><br>**—**", unsafe_allow_html=True)
            with c5: st.metric("Draw", f"{dr:.0f}%")
            fig_t, ax_t = plt.subplots(figsize=(8, 0.6))
            fig_t.patch.set_facecolor('#0e1117'); ax_t.set_facecolor('#0e1117')
            ax_t.barh([0],[hw], color='#00d4a4')
            ax_t.barh([0],[dr], left=[hw], color='#888888')
            ax_t.barh([0],[aw], left=[hw+dr], color='#FF6B6B')
            ax_t.set_xlim(0,100); ax_t.set_yticks([])
            ax_t.spines[:].set_visible(False)
            plt.tight_layout(pad=0); st.pyplot(fig_t)
            st.markdown("---")
    else:
        next_fixtures = [(ta,tb,grp,d) for ta,tb,grp,d in WC_FIXTURES if d >= today_str]
        if next_fixtures:
            next_date = next_fixtures[0][3]
            next_day  = [(ta,tb,grp,d) for ta,tb,grp,d in WC_FIXTURES if d == next_date]
            nd = datetime.strptime(next_date, "%Y-%m-%d").strftime("%B %d, %Y")
            st.info(f"🗓️ No matches today. Next fixtures: **{nd}**")
            st.markdown(f"### Upcoming — {nd}")
            for ta, tb, grp, d in next_day:
                hw, dr, aw = get_live_pred(ta, tb)
                c1,c2,c3,c4 = st.columns([3,1,3,2])
                with c1: st.markdown(f"**{ta}**  `{hw*100:.0f}%`")
                with c2: st.markdown("**vs**")
                with c3: st.markdown(f"**{tb}**  `{aw*100:.0f}%`")
                with c4: st.markdown(f"Group {grp} | Draw: {dr*100:.0f}%")
        else:
            st.success("🏆 Tournament complete!")

    if tomorrows:
        st.markdown("---")
        st.markdown("### 📆 Tomorrow")
        for ta, tb, grp, d in tomorrows:
            hw, dr, aw = get_live_pred(ta, tb)
            c1,c2,c3,c4 = st.columns([3,1,3,2])
            with c1: st.markdown(f"**{ta}**  `{hw*100:.0f}%`")
            with c2: st.markdown("**vs**")
            with c3: st.markdown(f"**{tb}**  `{aw*100:.0f}%`")
            with c4: st.markdown(f"Group {grp} | Draw: {dr*100:.0f}%")

# ══════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY
# ══════════════════════════════════════════════════════════════
elif page == "📖 Methodology":
    st.subheader("📖 How WorldSim Works")
    st.markdown("---")
    st.markdown("### 🗄️ Data")
    st.markdown("""
WorldSim is trained on **49,287 international football matches** from 1990 to 2026,
sourced from Kaggle's international football results dataset. We filter to competitive
matches only — World Cup, European Championship, Copa América, and qualifiers —
giving us **10,782 high-signal matches** for model fitting.
    """)
    st.markdown("---")
    st.markdown("### ⚡ Elo Rating Engine")
    st.markdown("""
Every team starts with an Elo rating of **1500**. After each match, ratings update using:

```
new_rating = old_rating + K × (actual − expected)
```

Where:
- **K = 32** (update speed)
- **expected** = 1 / (1 + 10^((opponent_elo − team_elo) / 400))
- **actual** = 1 (win), 0.5 (draw), 0 (loss)

After 32,101 matches the pre-tournament top ratings are:
**Spain 2041 | Argentina 2035 | France 1995 | England 1919 | Brazil 1922**

During the tournament, Elo ratings update live after each result you enter.
    """)
    st.markdown("---")
    st.markdown("### 📐 Dixon-Coles Goal Model")
    st.markdown("""
The Dixon-Coles model treats goals as independent Poisson processes.
Model fitted via **maximum likelihood** (L-BFGS-B + Powell two-stage optimisation).

Key fitted parameters:
- **Home advantage**: 1.295× (~30% goal boost)
- **ρ (rho)**: −0.059 (low-score correction)
- **Log-likelihood**: −7,043.22 (converged ✅)
    """)
    st.markdown("---")
    st.markdown("### 🔄 Live Bayesian Updates")
    st.markdown("""
When you enter a match result in **🔄 Update Results**:

1. The winning team gains Elo points (proportional to upset magnitude)
2. The losing team loses the same points
3. All future match predictions use the updated Elo ratings
4. This means a Spain loss would immediately reduce their predicted win probability

The match predictor and today's fixtures both use live Elo — so probabilities
reflect the current state of the tournament, not just pre-tournament predictions.
    """)
    st.markdown("---")
    st.markdown("### 🎲 Monte Carlo Simulation")
    st.markdown("""
WorldSim runs **10,000 complete tournament simulations**:

1. **Group stage** — full round-robin, 6 matches per group × 12 groups
2. **Best third-place selection** — FIFA rules: top 8 of 12 third-placed teams
3. **Round of 32 → Final** — knockout bracket with penalty shootout modelling

**Speed**: Pre-computed match cache enables 10,000 simulations in ~9 seconds.
    """)
    st.markdown("---")
    st.markdown("### 👤 Built by")
    st.markdown("""
**Abdul Razak Bilal** — B.Tech CSE (AI & ML), GPREC Kurnool

🔗 [LinkedIn](https://linkedin.com/in/abdul-razak-bilal) |
🔗 [GitHub](https://github.com/abdulrazakbilal) |
🔗 [Portfolio](https://abdulrazakbilal.github.io)
    """)

# ══════════════════════════════════════════════════════════════
# PAGE: BRACKET
# ══════════════════════════════════════════════════════════════
elif page == "🏟️ Bracket":
    st.subheader("🏟️ Predicted Group Stage Outcomes")
    st.markdown("Based on simulation results — uses live Elo where available.")

    predicted_bracket = {}
    for grp, teams in WC2026_GROUPS.items():
        grp_data = results_df[results_df['team'].isin(teams)]\
                   .sort_values('group_qualify%', ascending=False)
        if len(grp_data) >= 2:
            predicted_bracket[grp] = {
                'winner':    grp_data.iloc[0]['team'],
                'runner_up': grp_data.iloc[1]['team'],
                'w_qual':    grp_data.iloc[0]['group_qualify%'],
                'r_qual':    grp_data.iloc[1]['group_qualify%'],
            }

    group_cols = st.columns(3)
    for i, (grp, data) in enumerate(predicted_bracket.items()):
        with group_cols[i % 3]:
            st.markdown(f"**Group {grp}**")
            st.markdown(f"🥇 {data['winner']} `{data['w_qual']:.0f}%`")
            st.markdown(f"🥈 {data['runner_up']} `{data['r_qual']:.0f}%`")
            st.markdown("")

    st.markdown("---")
    st.markdown("### Predicted Round of 32 Fixtures")
    grp_keys = sorted(predicted_bracket.keys())

    r32_fixtures = []
    for i, grp in enumerate(grp_keys[:8]):
        home = predicted_bracket[grp]['winner']
        away_grp = grp_keys[(i + 6) % 12]
        away = predicted_bracket[away_grp]['runner_up']
        r32_fixtures.append((home, away, f"Group {grp} W vs Group {away_grp} R"))
    for i, grp in enumerate(grp_keys[8:]):
        home = predicted_bracket[grp]['winner']
        away_grp = grp_keys[i]
        away = predicted_bracket[away_grp]['runner_up']
        r32_fixtures.append((home, away, f"Group {grp} W vs Group {away_grp} R"))

    col_left, col_right = st.columns(2)
    for i, (home, away, label) in enumerate(r32_fixtures[:16]):
        hw, dr, aw = get_live_pred(home, away)
        hw *= 100; dr *= 100; aw *= 100
        winner = home if hw > aw else away
        col = col_left if i % 2 == 0 else col_right
        with col:
            st.markdown(f"**{label}**")
            fig_b, ax_b = plt.subplots(figsize=(5, 0.4))
            fig_b.patch.set_facecolor('#0e1117'); ax_b.set_facecolor('#0e1117')
            ax_b.barh([0],[hw], color='#00d4a4')
            ax_b.barh([0],[dr], left=[hw], color='#555')
            ax_b.barh([0],[aw], left=[hw+dr], color='#FF6B6B')
            ax_b.set_xlim(0,100); ax_b.set_yticks([])
            ax_b.spines[:].set_visible(False)
            plt.tight_layout(pad=0); st.pyplot(fig_b)
            st.markdown(
                f"<span style='color:#00d4a4'>**{home}** {hw:.0f}%</span> | "
                f"Draw {dr:.0f}% | "
                f"<span style='color:#FF6B6B'>**{away}** {aw:.0f}%</span> "
                f"→ *{winner} favoured*",
                unsafe_allow_html=True
            )
            st.markdown("")