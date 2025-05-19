import streamlit as st
import pandas as pd

# Load CSV
dataframe = pd.read_csv("/Users/omardukureh12/Omar_Dukureh_Capstone_Final/outputs/figures/WS_compare.csv")
dataframe.columns = dataframe.columns.str.strip()  # Clean headers

player = st.sidebar.selectbox("Select a Player", dataframe['Player'].unique())


# Streamlit title
st.title("🏀 Player Win Shares Dashboard")
st.header(f"⛹️‍♂️ {player}'s Win Shares")

# Filter selected player's row
player_row = dataframe[dataframe['Player'] == player]

# Safety check
if player_row.empty:
    st.warning("No data found for this player.")
else:
    # Extract values
    ws_2025 = player_row.iloc[0].get('WS', None)
    ws_2026 = player_row.iloc[0].get('WS_pred', None)
    diff = player_row.iloc[0].get('WS_diff', None)

    # Handle missing data
    if pd.isna(ws_2025) or pd.isna(ws_2026) or pd.isna(diff):
        st.warning("Some data is missing for this player.")
    else:
        # Determine color
        if diff >= 1:
            color = "green"
        elif diff <= -1:
            color = "red"
        else:
            color = "yellow"

        # Use columns for layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="2025 Win Shares", value=f"{ws_2025:.2f}")

        with col2:
            st.metric(label="2026 Win Shares", value=f"{ws_2026:.2f}")

        with col3:
            st.markdown(f"<div style='font-weight:bold; font-size:16px; margin-bottom:4px;'>Win Share Diff</div>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='color:{color}; margin-top:0;'>{diff:.2f}</h2>", unsafe_allow_html=True)

