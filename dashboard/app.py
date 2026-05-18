import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

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

    def elo_win_prob(home, away):
        if home not in tf.index or away not in tf.index:
            return 0.4, 0.2, 0.4
        elo_h  = tf.loc[home, 'elo']
        elo_a  = tf.loc[away, 'elo']
        form_h = tf.loc[home, 'form']
        form_a = tf.loc[away, 'form']
        exp_h  = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        form_adj = ((form_h + 0.5) / (form_a + 0.5)) ** 0.15
        exp_h  = float(np.clip(exp_h * form_adj, 0.05, 0.95))
        draw   = float(np.clip(0.28 * (1 - abs(exp_h - 0.5) * 1.2), 0.18, 0.30))
        win_h  = exp_h * (1 - draw)
        win_a  = (1 - exp_h) * (1 - draw)
        return win_h, draw, win_a

    cache_knockout = {}
    for home in all_wc_teams:
        for away in all_wc_teams:
            if home != away:
                w, d, l = elo_win_prob(home, away)
                cache_knockout[(home, away)] = {
                    'home_win': w, 'draw': d, 'away_win': l,
                    'mu_h': max(w*1.5, 0.3), 'mu_a': max(l*1.5, 0.3)
                }
    return cache_knockout, tf

results_df, team_features, model_data = load_data()

with st.spinner("⚙️ Building match probability cache..."):
    match_cache_knockout, team_features = build_match_cache(team_features)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("🏆 WorldSim 2026")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "⚔️ Match Predictor",
    "🗂️ Group Explorer",
    "📈 Path Probabilities",
    "🔄 Update Results"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Elo + Form fusion")
st.sidebar.markdown("- Dixon-Coles goal model")
st.sidebar.markdown("- 10,000 Monte Carlo sims")
st.sidebar.markdown("- Updated: May 2026")

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
    st.markdown("Select any two teams to see win probabilities and tournament path comparison.")

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
        pred = match_cache_knockout.get((home_team, away_team))
        if pred:
            hw = pred['home_win'] * 100
            dr = pred['draw']     * 100
            aw = pred['away_win'] * 100

            st.markdown(f"### {home_team}  vs  {away_team}")
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
            ax2.set_xlim(0, 100)
            ax2.set_yticks([])
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
                h_row = h_row.iloc[0]
                a_row = a_row.iloc[0]
                stages = ['reach_R16%','reach_QF%','reach_SF%','reach_Final%','champion%']
                labels = ['R16','QF','SF','Final','🏆']
                fig3, ax3 = plt.subplots(figsize=(10, 4))
                fig3.patch.set_facecolor('#0e1117')
                ax3.set_facecolor('#1e2130')
                x = np.arange(len(labels))
                w = 0.35
                ax3.bar(x-w/2, [h_row[s] for s in stages], w,
                       label=home_team, color='#00d4a4')
                ax3.bar(x+w/2, [a_row[s] for s in stages], w,
                       label=away_team, color='#FF6B6B')
                ax3.set_xticks(x)
                ax3.set_xticklabels(labels, color='white', fontsize=12)
                ax3.set_ylabel('Probability (%)', color='white')
                ax3.set_title('Tournament Path Comparison',
                             color='white', fontweight='bold')
                ax3.legend(facecolor='#1e2130', labelcolor='white')
                ax3.tick_params(colors='white')
                ax3.spines[:].set_color('#444')
                ax3.grid(axis='y', alpha=0.2, color='white')
                plt.tight_layout()
                st.pyplot(fig3)
        else:
            st.warning("Match data not available.")

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
    fig4.patch.set_facecolor('#0e1117')
    ax4.set_facecolor('#1e2130')
    teams = group_data['team'].tolist()
    x = np.arange(len(teams))
    w = 0.25
    ax4.bar(x-w, group_data['group_qualify%'].values, w,
            label='Qualify', color='#00d4a4')
    ax4.bar(x,   group_data['reach_QF%'].values,      w,
            label='Reach QF', color='#FFD700')
    ax4.bar(x+w, group_data['champion%'].values,      w,
            label='Champion', color='#FF6B6B')
    ax4.set_xticks(x)
    ax4.set_xticklabels(teams, color='white', fontsize=11)
    ax4.set_ylabel('Probability (%)', color='white')
    ax4.set_title(f'Group {group_letter} — Team Probabilities',
                 color='white', fontweight='bold', fontsize=13)
    ax4.legend(facecolor='#1e2130', labelcolor='white')
    ax4.tick_params(colors='white')
    ax4.spines[:].set_color('#444')
    ax4.grid(axis='y', alpha=0.2, color='white')
    plt.tight_layout()
    st.pyplot(fig4)

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
    st.markdown("Track any team's probability at each stage.")

    selected_team = st.selectbox("Select Team", results_df['team'].tolist())
    row    = results_df[results_df['team']==selected_team].iloc[0]
    stages = ['group_qualify%','reach_R16%','reach_QF%',
              'reach_SF%','reach_Final%','champion%']
    labels = ['Qualify','R16','QF','SF','Final','🏆 Win']
    values = [row[s] for s in stages]

    cols = st.columns(6)
    for i, (lbl, val) in enumerate(zip(labels, values)):
        with cols[i]: st.metric(lbl, f"{val:.1f}%")

    fig5, ax5 = plt.subplots(figsize=(12, 5))
    fig5.patch.set_facecolor('#0e1117')
    ax5.set_facecolor('#1e2130')
    ax5.plot(labels, values, color='#00d4a4',
             linewidth=2.5, marker='o', markersize=8)
    ax5.fill_between(range(len(labels)), values, alpha=0.15, color='#00d4a4')
    for i, val in enumerate(values):
        ax5.annotate(f'{val:.1f}%', (i, val),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', color='white', fontsize=10)
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(labels, color='white', fontsize=11)
    ax5.set_ylabel('Probability (%)', color='white')
    ax5.set_title(f'{selected_team} — Tournament Path Probabilities',
                 color='white', fontweight='bold', fontsize=13)
    ax5.tick_params(colors='white')
    ax5.spines[:].set_color('#444')
    ax5.grid(alpha=0.2, color='white')
    plt.tight_layout()
    st.pyplot(fig5)

# ══════════════════════════════════════════════════════════════
# PAGE: UPDATE RESULTS
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Update Results":
    st.subheader("🔄 Live Match Result Entry")
    st.markdown("Enter real match results as the tournament progresses. "
                "The model will show how probabilities would shift.")
    st.info("🗓️ Tournament starts June 11, 2026 — Group Stage Day 1")

    st.markdown("#### Enter a match result")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        team_a = st.selectbox("Home Team", all_wc_teams,
                               index=all_wc_teams.index('Mexico'))
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### vs")
    with col3:
        team_b = st.selectbox("Away Team", all_wc_teams,
                               index=all_wc_teams.index('South Africa'))

    col4, col5 = st.columns(2)
    with col4:
        score_a = st.number_input(f"{team_a} goals",
                                   min_value=0, max_value=20, value=0)
    with col5:
        score_b = st.number_input(f"{team_b} goals",
                                   min_value=0, max_value=20, value=0)

    st.selectbox("Tournament stage", [
        "Group Stage", "Round of 32", "Round of 16",
        "Quarter Final", "Semi Final", "Final"
    ])

    if st.button("⚡ Analyse Result", type="primary"):
        st.markdown(f"### Result: {team_a} {int(score_a)} — {int(score_b)} {team_b}")

        if team_a in team_features.index and team_b in team_features.index:
            elo_a = team_features.loc[team_a, 'elo']
            elo_b = team_features.loc[team_b, 'elo']

            exp_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

            if score_a > score_b:
                actual_a, actual_b = 1.0, 0.0
                result_str = f"✅ {team_a} WIN"
            elif score_b > score_a:
                actual_a, actual_b = 0.0, 1.0
                result_str = f"✅ {team_b} WIN"
            else:
                actual_a, actual_b = 0.5, 0.5
                result_str = "🤝 DRAW"

            k = 32
            new_elo_a = elo_a + k * (actual_a - exp_a)
            new_elo_b = elo_b + k * (actual_b - (1 - exp_a))
            delta_a = new_elo_a - elo_a
            delta_b = new_elo_b - elo_b

            st.markdown(f"**{result_str}**")
            st.markdown("---")
            st.markdown("#### ⚡ Elo Rating Changes")

            c1, c2 = st.columns(2)
            with c1:
                st.metric(team_a, f"{new_elo_a:.0f}", f"{delta_a:+.1f}")
                st.markdown(f"*Was {elo_a:.0f} → Now {new_elo_a:.0f}*")
            with c2:
                st.metric(team_b, f"{new_elo_b:.0f}", f"{delta_b:+.1f}")
                st.markdown(f"*Was {elo_b:.0f} → Now {new_elo_b:.0f}*")

            # Upset analysis
            pre_match = match_cache_knockout.get((team_a, team_b))
            if pre_match:
                model_fav = team_a if pre_match['home_win'] > pre_match['away_win'] \
                            else team_b
                if score_a > score_b:
                    actual_winner = team_a
                elif score_b > score_a:
                    actual_winner = team_b
                else:
                    actual_winner = "Draw"

                st.markdown("---")
                st.markdown("#### 🚨 Upset Analysis")

                if actual_winner != "Draw" and actual_winner != model_fav:
                    upset_prob = pre_match['away_win'] if actual_winner == team_b \
                                 else pre_match['home_win']
                    st.error(f"⚠️ UPSET DETECTED! Model gave {actual_winner} only "
                            f"{upset_prob*100:.1f}% chance of winning.")
                elif actual_winner == "Draw":
                    st.info(f"Draw — model gave {pre_match['draw']*100:.1f}% "
                           f"probability to this outcome.")
                else:
                    fav_prob = pre_match['home_win'] if model_fav == team_a \
                               else pre_match['away_win']
                    st.success(f"✅ Favourite won as expected "
                              f"({fav_prob*100:.1f}% pre-match probability).")

            st.markdown("---")
            st.info("💡 After the group stage ends, re-run the Monte Carlo simulation "
                   "with updated Elo ratings to refresh all championship probabilities.")
        else:
            st.warning("Team data not available for Elo calculation.")

    st.markdown("---")
    st.markdown("#### 📋 Opening Day Fixtures — June 11-12")
    day1 = [
        ("Mexico",         "South Africa", "Group A", "Jun 11"),
        ("United States",  "Paraguay",     "Group D", "Jun 12"),
        ("Canada",         "Bosnia and Herzegovina", "Group B", "Jun 12"),
        ("Germany",        "Curacao",      "Group E", "Jun 12"),
    ]
    for ta, tb, grp, date in day1:
        pred = match_cache_knockout.get((ta, tb))
        if pred:
            c1, c2, c3, c4 = st.columns([2, 1, 2, 3])
            with c1: st.markdown(f"**{ta}**")
            with c2: st.markdown("vs")
            with c3: st.markdown(f"**{tb}**")
            with c4:
                st.markdown(f"{grp} | {date} | "
                           f"{ta}: {pred['home_win']*100:.0f}% — "
                           f"{tb}: {pred['away_win']*100:.0f}%")