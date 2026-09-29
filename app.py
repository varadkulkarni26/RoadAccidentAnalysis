"""
app.py
------
Streamlit dashboard for Indian Road Accident Severity Analysis.

Run with:
    streamlit run app.py

Expects data/india_road_accidents_clean.csv produced by data_cleaning.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib

st.set_page_config(
    page_title="India Road Accident Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main, .block-container {padding-top:1.6rem; background:#151623;}
    [data-testid="stSidebar"] {background: #181b2a !important;}
    .stMetric {padding-bottom:18px;}
    .stMetric-label {color: #36d399;}
    </style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "india_road_accidents_clean.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

st.sidebar.title(" India Road Accident Dashboard")
section = st.sidebar.radio(
    "Menu",
    ["National Trends", "State Trends", "Time & Season Analysis",
     "Vehicle Type Analysis", "Severity Prediction"],
)

# ---------------- National Trends ----------------
if section == "National Trends":
    st.title("🇮🇳 National Accident Trends")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Accidents", f"{len(df):,}")
    if "severity_clean" in df.columns:
        fatal_count = (df["severity_clean"] == "fatal").sum()
        c2.metric("Fatal Accidents", f"{fatal_count:,}")
        c3.metric("Fatality Rate", f"{fatal_count / len(df) * 100:.1f}%")

    if "year" in df.columns:
        yearly = df.groupby("year").size().reset_index(name="count")
        fig = px.line(yearly, x="year", y="count", markers=True,
                       title="Accidents per Year")
        st.plotly_chart(fig, use_container_width=True)

    if "severity_clean" in df.columns:
        sev_counts = df["severity_clean"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig2 = px.bar(sev_counts, x="severity", y="count",
                       title="Accidents by Severity", color="severity")
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(df.head(50), use_container_width=True)
    st.download_button("Download Full Dataset (CSV)",
                        df.to_csv(index=False), "accidents.csv")

# ---------------- State Trends ----------------
elif section == "State Trends":
    st.title(" State-wise Accident Trends")

    if "state" in df.columns:
        state_counts = df.groupby("state").size().reset_index(name="count")
        state_counts = state_counts.sort_values("count", ascending=False)

        top_n = st.sidebar.slider("Show top N states", 5, 30, 10)
        fig = px.bar(state_counts.head(top_n), x="state", y="count",
                     title=f"Top {top_n} States by Accident Count",
                     color="count", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        # Choropleth needs a GeoJSON of Indian state boundaries.
        # Download one (e.g. from https://github.com/geohacker/india) and
        # point GEOJSON_PATH to it to enable the map below.
        GEOJSON_PATH = os.path.join("data", "india_states.geojson")
        if os.path.exists(GEOJSON_PATH):
            import json
            with open(GEOJSON_PATH) as f:
                india_geo = json.load(f)
            fig_map = px.choropleth(
                state_counts, geojson=india_geo,
                featureidkey="properties.NAME_1",  # adjust to your geojson's key
                locations="state", color="count",
                color_continuous_scale="OrRd",
                title="Accident Density by State",
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info(
                "Add an India states GeoJSON at data/india_states.geojson "
                "to enable the choropleth map. Falling back to bar chart above."
            )

        st.dataframe(state_counts, use_container_width=True)
        st.download_button("Download State Summary (CSV)",
                            state_counts.to_csv(index=False), "state_summary.csv")

# ---------------- Time & Season Analysis ----------------
elif section == "Time & Season Analysis":
    st.title(" Time-of-Day & Seasonal Patterns")

    if "hour" in df.columns and df["hour"].notna().any():
        hourly = df.groupby("hour").size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count",
                     title="Accidents by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

    if "season" in df.columns:
        season_counts = df.groupby("season").size().reset_index(name="count")
        fig2 = px.pie(season_counts, names="season", values="count",
                       title="Accidents by Season")
        st.plotly_chart(fig2, use_container_width=True)

    if "is_weekend" in df.columns:
        weekend_counts = df.groupby("is_weekend").size().reset_index(name="count")
        weekend_counts["is_weekend"] = weekend_counts["is_weekend"].map(
            {True: "Weekend", False: "Weekday"})
        fig3 = px.bar(weekend_counts, x="is_weekend", y="count",
                       title="Weekday vs Weekend Accidents")
        st.plotly_chart(fig3, use_container_width=True)

# ---------------- Vehicle Type Analysis ----------------
elif section == "Vehicle Type Analysis":
    st.title(" Vehicle Type Analysis")

    if "vehicle_type" in df.columns:
        vt = df.groupby("vehicle_type").size().reset_index(name="count")
        vt = vt.sort_values("count", ascending=False)
        fig = px.bar(vt, x="vehicle_type", y="count",
                     title="Accidents by Vehicle Type", color="count")
        st.plotly_chart(fig, use_container_width=True)

        if "severity_clean" in df.columns:
            cross = pd.crosstab(df["vehicle_type"], df["severity_clean"])
            fig2 = px.imshow(cross, text_auto=True, aspect="auto",
                              title="Vehicle Type vs Severity (counts)")
            st.plotly_chart(fig2, use_container_width=True)

# ---------------- Severity Prediction ----------------
elif section == "Severity Prediction":
    st.title(" Accident Severity Prediction")
    st.write(
        "Select conditions below and the trained model will predict the "
        "most likely severity class. Run severity_classification.py first "
        "to generate the model file."
    )

    MODEL_PATH = os.path.join("models", "severity_model.pkl")
    ENCODERS_PATH = os.path.join("models", "encoders.pkl")

    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)

        col1, col2 = st.columns(2)
        state_val = col1.selectbox("State", sorted(df["state"].dropna().unique()))
        vehicle_val = col2.selectbox(
            "Vehicle Type",
            sorted(df["vehicle_type"].dropna().unique())
            if "vehicle_type" in df.columns else ["Unknown"],
        )
        hour_val = st.slider("Hour of Day", 0, 23, 12)
        num_vehicles_val = st.slider("Number of Vehicles Involved", 1, 6, 2)

        if st.button("Predict Severity"):
            input_df = pd.DataFrame([{
                "state": encoders["state"].transform([state_val])[0],
                "vehicle_type": encoders["vehicle_type"].transform([vehicle_val])[0],
                "hour": hour_val,
                "num_vehicles": num_vehicles_val,
            }])
            pred = model.predict(input_df)[0]
            pred_label = encoders["severity"].inverse_transform([pred])[0]
            st.success(f"Predicted Severity: **{pred_label.title()}**")
    else:
        st.warning("No trained model found. Run severity_classification.py first.")

st.markdown(
    """
    <div style='position: fixed; right: 24px; bottom: 16px; font-size: 12px; color: #888;'>
    India Road Accident Analysis — Mini Project
    </div>
    """,
    unsafe_allow_html=True,
)