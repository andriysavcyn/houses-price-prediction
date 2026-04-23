import pandas as pd
import streamlit as st
from streamlit_lottie import st_lottie
import joblib 
import requests

#1
st.set_page_config("House Price Prediction", layout="wide", page_icon="🏙️")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .main-title { font-size: 3rem; font-weight: 700; color: #FF4B4B; text-align: center; }
    </style>
    """, unsafe_allow_html=True)
#2
@st.cache_resource
def load_lottie(url):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load("notebooks/full_pipeline.pkl")
        model = joblib.load("notebooks/final_model.pkl")
        return pipeline, model
    except Exception as e:
        st.error(f":Error loading models: {e}")
        return None, None


# header animation
lottie_house = load_lottie(
    "https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json"
)
pipeline, model = load_models()

st.sidebar.header("District characteristics")

with st.sidebar:
    st.subheader("Geography")
    lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value = 37.7, step=0.1, key="latitude")
    lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value = -122.4, step=0.1, key="longitude")
    ocean_type = st.sidebar.selectbox("Ocean proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])
    
    st.subheader("Demographics")
    income = st.sidebar.number_input("Median income ($10k units)", min_value=0.0, value = 5.0, step=2.0, key="income")
    age = st.sidebar.number_input("House age", min_value=0.0, value = 29.0, step=2.0, key="age")
    pop = st.sidebar.number_input("Population", min_value=0.0, step=100.0, key="population")
    hh = st.sidebar.number_input("Number of households", min_value=0.0, value = 29.0, step=2.0, key="households")

    st.subheader("Property Details")
    rooms = st.sidebar.number_input("Number of rooms", min_value=0.0, step=100.0, key="rooms")
    bedrooms = st.sidebar.number_input("Number of bedrooms", min_value=0.0, step=100.0, key="bedrooms")
#4
c1, c2 = st.columns([1,4])

with c1:
    if lottie_house: st_lottie(lottie_house, height=120)
with c2:
    st.markdown('<p class="main-title">California House Price Estimator</p>', unsafe_allow_html=True)
    st.caption("Machine Learning model trained on 1990 California Census data to predict median district house values.")

st.divider()
#5
if st.button("Estimate property value:", use_container_width=True):
    if pipeline and model:
        user_data = {
            "median_income": income,
            "housing_median_age": age,
            "total_rooms": rooms,
            "total_bedrooms": bedrooms,
            "population": pop,
            "households": hh,
            "latitude": lat,
            "longitude": lon,
            "ocean_proximity": ocean_type
        }
        df = pd.DataFrame([user_data])

        df["rooms_per_household"] = df["total_rooms"] / df["households"]
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
        df["population_per_household"] = df["population"] / df["households"]

        with st.spinner('AI is analyzing district data...'):
            X_prepared = pipeline.transform(df)
            prediction = model.predict(X_prepared)[0]

#6
        res_col1, res_col2 = st.columns([1, 1.5])


        with res_col1:
            st.success(f"### Estimated Price: ${prediction:,.2f}")

            st.metric("Income Comparison", f"{income}", f"{income-3.8:+.2f} vs Avg")
            st.metric("House Age", f"{int(age)} years", f"{int(age-28):+d} vs Avg", delta_color="inverse")

            with st.expander("🛠️ View Engineered Features"):
                st.write("Computed values used by the model:")
                st.dataframe(df[["rooms_per_household", "bedrooms_per_room", "population_per_household"]].T)

        with res_col2:
            st.subheader("📍 Location")
            st.map(df, size = 40, zoom = 10)

    else:
        st.error("Model not available. Please check system logs.")
#7
st.divider()
st.markdown(f"""<div style="text-align: center; color: grey; font-size: 0.8rem;">
        Developed by Andriy Savchyn | Lviv | 2026<br>
        <a href="https://github.com/andriysavcyn" target="_blank">GitHub</a> • 
        <a href="https://www.linkedin.com/in/andriy-savchyn" target="_blank">LinkedIn</a>
    </div>""", unsafe_allow_html=True)

