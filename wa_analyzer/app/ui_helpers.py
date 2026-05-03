"""Shared UI/chart helpers for the Streamlit app."""

import pandas as pd


# --- Helper: Calculate Correlation Matrix Text ---
def get_correlation_text(df):
    """
    Calculate Pearson correlations between columns of a DataFrame.
    Returns a formatted string showing correlation pairs.
    """
    if df is None or df.empty or len(df.columns) < 2:
        return None

    # Drop rows with NaN to get valid correlation
    df_clean = df.dropna()
    if len(df_clean) < 3:  # Need at least 3 data points
        return None

    cols = df.columns.tolist()
    correlations = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = df_clean[cols[i]].corr(df_clean[cols[j]])
            if pd.notna(corr):
                # Format: positive = move together, negative = opposite
                direction = "↗↗" if corr > 0.5 else "↗↘" if corr < -0.5 else "→"
                correlations.append(
                    f"**{cols[i]}** vs **{cols[j]}**: r={corr:.2f} {direction}"
                )

    return " | ".join(correlations) if correlations else None


