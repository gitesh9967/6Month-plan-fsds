# app.py
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Mall Customers", layout="wide")

# ---- Minimal styling (thin top gradient like your screenshot) ----
st.markdown("""
<style>
.topbar {
  height: 4px; background: linear-gradient(90deg,#ff4d4d,#ffd11a);
  position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
}
.small-muted { color:#6b7280; font-size:0.95rem; }
.result-badge {
  display:inline-block; padding:14px 20px; border-radius:12px;
  border:1px solid #e5e7eb; font-weight:600;
}
</style>
<div class="topbar"></div>
""", unsafe_allow_html=True)

# ------------------ MODEL LOADING ------------------
st.sidebar.write("### Load Model")
model = None
status = ""

# Try local file named exactly "mall_pkl"
try:
    with open(r"C:\Users\rosha\OneDrive\Desktop\Gitesh\NIT\TOPICS\ML\zzzzztypes\Mall Project\mall_pkl", "rb") as f:
        model = pickle.load(f)
        status = "Loaded local model: mall_pkl"
except Exception as e:
    status = f"Local model not found ({e}). You can upload below."

uploaded = st.sidebar.file_uploader("…or upload a .pkl", type=["pkl"])
if uploaded:
    try:
        model = pickle.load(uploaded)
        status = "Loaded uploaded .pkl"
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")

st.sidebar.info(status)

if model is None:
    st.stop()

# ------------------ LAYOUT ------------------
left, right = st.columns([1, 3])

with right:
    st.markdown("## Mall Customers")
    st.markdown(
        "<div class='small-muted'>This app divides customers into clusters using a K-Means model.</div>",
        unsafe_allow_html=True
    )
    st.write("")

with left:
    st.write("#### Age")
    age = st.number_input("", min_value=0, max_value=120, value=25, step=1, key="age")

    st.write("#### Annual income (k$)")
    income = st.number_input("", min_value=0.0, max_value=10000.0, value=1000.0, step=10.0, key="inc")

    st.write("#### Spending score (1-100)")
    score = st.number_input("", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="score")

    predict_btn = st.button("Type of Customers")

    st.write("")
    st.selectbox("Select Origin:", ["VIP"], index=0, disabled=True)

# ------------------ PREDICTION ------------------
with right:
    if predict_btn:
        try:
            X = np.array([[age, income, score]], dtype=float)

            # Some users save a pipeline (scaler + kmeans). Some save raw KMeans.
            # Either way, .predict should work if feature order matches training.
            y = model.predict(X)
            cluster_id = int(y[0])

            # Optional simple naming (tweak if you have your own)
            names = {
                0: "General",
                1: "VIP",
                2: "Careful",
                3: "Target",
                4: "Sensible",
            }
            label = names.get(cluster_id, f"Cluster {cluster_id}")

            st.markdown(f"### Result")
            st.markdown(f"<span class='result-badge'>Customer Type: {label}</span>", unsafe_allow_html=True)
            st.caption(f"Raw cluster id: {cluster_id}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
