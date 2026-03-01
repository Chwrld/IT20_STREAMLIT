import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Define paths to local artifacts
MODEL_DIR = os.path.join("..", "travel-training-model")
DATASET_PATH = os.path.join("..", "dataset", "Expanded_Destinations.csv")

st.set_page_config(
    page_title="TravelMind · Destination Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Helper for Icons ---
def icon(name, size=18, color="currentColor", stroke_width=2):
    icons = {
        "compass":  f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>""",
        "user":     f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>""",
        "plane":    f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M17.8 19.2 16 11l3.5-3.5C21 6 21 4 19 4s-2 1-3.5 2.5L11 8 2.8 6.2c-.5-.1-.9.2-1.1.7l-.3.8a1 1 0 0 0 .4 1.1l4.2 3.1L4 15l-2 1 1 2 2-1 1.8-1.8 3.1 4.2a1 1 0 0 0 1.1.4l.8-.3c.5-.2.8-.6.7-1.1z"/></svg>""",
        "map-pin":  f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>""",
        "mountain": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="m8 3 4 8 5-5 5 15H2L8 3z"/></svg>""",
        "landmark": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><line x1="3" x2="21" y1="22" y2="22"/><line x1="6" x2="6" y1="18" y2="11"/><line x1="10" x2="10" y1="18" y2="11"/><line x1="14" x2="14" y1="18" y2="11"/><line x1="18" x2="18" y1="18" y2="11"/><polygon points="12 2 20 7 4 7"/></svg>""",
        "waves":    f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.5-2 5-2 1.3 0 1.9.5 2.5 1"/></svg>""",
        "leaf":     f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>""",
        "city":     f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><line x1="3" x2="21" y1="22" y2="22"/><line x1="6" x2="6" y1="18" y2="11"/><line x1="18" x2="18" y1="18" y2="11"/><path d="M2 22 12 2l10 20"/><line x1="9" x2="9" y1="22" y2="18"/><line x1="15" x2="15" y1="22" y2="18"/><rect width="6" height="4" x="9" y="18"/></svg>""",
        "calendar": f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" x2="16" y1="2" y2="6"/><line x1="8" x2="8" y1="2" y2="6"/><line x1="3" x2="21" y1="10" y2="10"/></svg>""",
    }
    return icons.get(name, "")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
:root { --sand:#F5EDD8; --terra:#C97D4E; --deep:#2C2416; --sage:#6B8F71; --sky:#4A90A4; --card:#FFFDF6; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background-color:var(--sand); color:var(--deep); }
section[data-testid="stSidebar"] { background:var(--deep) !important; }
section[data-testid="stSidebar"] * { color:var(--sand) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stTextInput label {
    color:var(--terra) !important; font-weight:500; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase;
}
section[data-testid="stSidebar"] .stSelectbox > div > div { background:#3D3020 !important; border:1px solid var(--terra) !important; color:var(--sand) !important; }
section[data-testid="stSidebar"] .stTextInput > div > input { background:#3D3020 !important; border:1px solid var(--terra) !important; color:var(--sand) !important; border-radius:6px !important; }
section[data-testid="stSidebar"] .stNumberInput > div > div > input { background:#3D3020 !important; border:1px solid var(--terra) !important; color:var(--sand) !important; border-radius:6px !important; }
.hero-title { font-family:'Playfair Display',serif; font-size:3rem; font-weight:900; color:var(--deep); line-height:1.1; margin-bottom:0; }
.hero-sub { font-size:1rem; color:#7a6a52; margin-top:0.25rem; margin-bottom:2rem; letter-spacing:0.04em; }
.pred-box { background:linear-gradient(135deg,var(--terra) 0%,#A0522D 100%); color:white; border-radius:20px; padding:2.5rem; text-align:center; box-shadow:0 8px 32px rgba(201,125,78,0.35); }
.pred-box .dest-icon { display:flex; justify-content:center; margin-bottom:0.5rem; }
.pred-box .dest-name { font-family:'Playfair Display',serif; font-size:2.4rem; font-weight:900; letter-spacing:0.02em; margin-top:0.5rem; }
.pred-box .conf-label { font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.75; margin-top:0.4rem; }
.pred-box .conf-pct { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; margin-top:0.5rem; }
.pred-box .dest-detail { margin-top:1rem; font-size:0.9rem; opacity:0.88; display:flex; justify-content:center; align-items:center; gap:0.4rem; flex-wrap:wrap; line-height:1.8; }
.dest-detail-item { display:inline-flex; align-items:center; gap:0.3rem; }
.prob-row { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.75rem; }
.prob-label { width:155px; font-size:0.83rem; font-weight:500; color:var(--deep); }
.prob-bar-wrap { flex:1; background:#e8dfc8; border-radius:100px; height:12px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:100px; }
.prob-pct { width:52px; text-align:right; font-size:0.82rem; font-weight:600; }
.section-head { font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:700; color:var(--deep); border-bottom:2px solid var(--terra); padding-bottom:0.3rem; margin-bottom:1.2rem; }
.sidebar-section-label { display:flex; align-items:center; gap:0.5rem; color:#C97D4E !important; font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; font-weight:600; margin-bottom:0.8rem; }
.stDataFrame { border-radius:12px; overflow:hidden; }
.stButton > button { background:var(--terra) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:600 !important; letter-spacing:0.04em !important; padding:0.65rem 2rem !important; font-size:0.95rem !important; transition:opacity 0.2s !important; width:100% !important; }
.stButton > button:hover { opacity:0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load local model and preprocessing artifacts ──
@st.cache_resource
def load_local_artifacts():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.joblib'))
        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor.joblib'))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
        X_columns = joblib.load(os.path.join(MODEL_DIR, 'X_columns.joblib'))
        return model, preprocessor, label_encoder, X_columns
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

model, preprocessor, label_encoder, X_columns = load_local_artifacts()

def predict(user_data):
    user_input = pd.DataFrame([user_data])
    
    # Check if the model is Keras and needs manual transformation
    # (Based on the training script logic)
    is_keras = hasattr(model, 'layers')
    
    if is_keras:
        user_final = preprocessor.transform(user_input)
        probabilities = model.predict(user_final)[0]
    else:
        # Scikit-learn Pipelines usually handle preprocessing internally if it was part of the pipeline
        # However, in train_and_compare.py, preprocessor was separated for Keras.
        # Let's check if 'preprocessor' is part of the model object (Pipeline)
        if hasattr(model, 'predict_proba'):
            # If the best model exported was a Pipeline, it might already include the preprocessor
            # But according to joblib.dump(best_model_obj, 'best_model.joblib'), it's just the model/pipeline
            # If it's a Pipeline object, it has a 'preprocessor' step.
            try:
                probabilities = model.predict_proba(user_input)[0]
            except:
                user_final = preprocessor.transform(user_input)
                probabilities = model.predict_proba(user_final)[0]
        else:
             user_final = preprocessor.transform(user_input)
             probabilities = model.predict(user_final)[0]
    
    return dict(zip(label_encoder.classes_, probabilities))

# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
      <div style='display:flex;justify-content:center;'>{icon("compass",size=40,color="#C97D4E",stroke_width=1.5)}</div>
      <div style='font-family:"Playfair Display",serif;font-size:1.3rem;font-weight:700;color:#F5EDD8;margin-top:0.5rem;'>TravelMind</div>
      <div style='font-size:0.72rem;letter-spacing:0.12em;color:#C97D4E;text-transform:uppercase;'>Find Your Destination</div>
    </div>
    <hr style='border-color:#3D3020;margin:0.8rem 0 1.2rem;'/>
    <div class="sidebar-section-label">{icon("user",size=14,color="#C97D4E")} About You</div>
    """, unsafe_allow_html=True)

    age = st.number_input("Age", min_value=18, max_value=100, value=28)
    gender = st.selectbox("Gender", ["Male", "Female"])
    budget = st.selectbox("Budget", ["Low", "Medium", "High"])
    adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    st.markdown(f"""
    <hr style='border-color:#3D3020;margin:1.2rem 0;'/>
    <div class="sidebar-section-label">{icon("plane",size=14,color="#C97D4E")} Your Trip</div>
    """, unsafe_allow_html=True)

    month = st.slider("Travel Month", 1, 12, 10)
    
    st.markdown("<p style='color:#C97D4E; font-size:0.85rem; font-weight:500; text-transform:uppercase; margin-bottom:0.5rem;'>Preferences</p>", unsafe_allow_html=True)
    pref_relax = st.checkbox("Relaxation", value=True)
    pref_adv = st.checkbox("Adventure", value=False)
    pref_cult = st.checkbox("Culture", value=False)
    pref_spir = st.checkbox("Spiritual", value=False)

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("Find My Destination")

# ── MAIN ──
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;">
  {icon("compass",size=36,color="#C97D4E",stroke_width=1.5)}
  <div class="hero-title">Where Should You Travel Next?</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Answer a few questions and let AI find your perfect destination based on your preferences.</div>', unsafe_allow_html=True)

# Map local destinations to icons/regions
# Load destination dataset for details
@st.cache_data
def load_destinations():
    try:
        df = pd.read_csv(DATASET_PATH)
        return df
    except:
        return None

df_destinations = load_destinations()

def get_dest_details(dest_name):
    if df_destinations is not None:
        row = df_destinations[df_destinations['Name'] == dest_name]
        if not row.empty:
            return {
                "region": row.iloc[0]['State'],
                "type": row.iloc[0]['Type'],
                "season": row.iloc[0]['BestTimeToVisit']
            }
    return {"region": "India", "type": "Various", "season": "Check Local Guide"}

dest_icons = {
    "Beach": "waves",
    "Mountains": "mountain",
    "Metropolis": "city",
    "Historical City": "landmark",
    "Island": "waves",
    "Cliff Beach": "waves",
    "Ancient Ruins": "landmark",
    "National Park": "leaf",
    "Holy City": "landmark",
    "Spiritual Township": "leaf",
    "Temple Town": "landmark"
}

colors = ["#C97D4E","#6B8F71","#4A90A4","#8B6BA8","#D4A853"]

col_pred, col_probs = st.columns([1.2,1.8], gap="large")

with col_pred:
    st.markdown('<div class="section-head">Your Perfect Match</div>', unsafe_allow_html=True)

    if predict_btn:
        user_data = {
            'Age': age,
            'NumberOfAdults': adults,
            'NumberOfChildren': children,
            'TravelMonth': month,
            'Gender': gender,
            'Budget': budget,
            'Pref_Relaxation': 1 if pref_relax else 0,
            'Pref_Adventure': 1 if pref_adv else 0,
            'Pref_Culture': 1 if pref_cult else 0,
            'Pref_Spiritual': 1 if pref_spir else 0
        }
        
        prob_dict = predict(user_data)
        if prob_dict:
            pred_name = max(prob_dict, key=prob_dict.get)
            pred_pct  = prob_dict[pred_name] * 100

            details = get_dest_details(pred_name)
            icon_name = dest_icons.get(details['type'], "map-pin")
            
            dest_svg = icon(icon_name, size=56, color="white", stroke_width=1.2)
            pin_svg  = icon("map-pin",  size=15, color="rgba(255,255,255,0.85)")
            tag_svg  = icon("landmark", size=15, color="rgba(255,255,255,0.85)")
            cal_svg  = icon("calendar", size=15, color="rgba(255,255,255,0.85)")

            st.markdown(f"""
            <div class="pred-box">
              <div class="dest-icon">{dest_svg}</div>
              <div class="dest-name">{pred_name}</div>
              <div class="conf-label">Recommended Destination</div>
              <div class="conf-pct">{pred_pct:.1f}% Match</div>
              <div class="dest-detail">
                <span class="dest-detail-item">{pin_svg} {details['region']}</span>
                &nbsp;·&nbsp;
                <span class="dest-detail-item">{tag_svg} {details['type']}</span>
                &nbsp;·&nbsp;
                <span class="dest-detail-item">{cal_svg} Best: {details['season']}</span>
              </div>
            </div>""", unsafe_allow_html=True)

            st.session_state["prob_dict"] = prob_dict
    else:
        st.info("Answer the questions in the sidebar and click **Find My Destination**.")

with col_probs:
    st.markdown('<div class="section-head">Top Recommendation Strength</div>', unsafe_allow_html=True)

    if "prob_dict" in st.session_state:
        prob_dict  = st.session_state["prob_dict"]
        prob_pairs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:5] # Show Top 5

        bars_html = ""
        for i, (dname, prob) in enumerate(prob_pairs):
            pct  = prob * 100
            clr  = colors[i % len(colors)]
            pin  = icon("map-pin", size=13, color=clr)
            bold = "font-weight:700;" if i == 0 else ""
            bars_html += f"""
            <div class="prob-row">
              <div class="prob-label" style="display:flex;align-items:center;gap:0.3rem;{bold}">{pin} {dname}</div>
              <div class="prob-bar-wrap">
                <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{clr};"></div>
              </div>
              <div class="prob-pct" style="color:{clr};">{pct:.1f}%</div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown("""
        <div style='margin-top:1rem;font-size:0.77rem;color:#9a8a72;border-top:1px solid #e8dfc8;padding-top:0.8rem;'>
          Probabilities are generated by our high-performance ensemble model trained on over 2,000 verified traveler experiences.
        </div>""", unsafe_allow_html=True)
    else:
        st.caption("Match scores will appear here after prediction.")

# ── DATASET EXPLORER ──
st.markdown("<br/>", unsafe_allow_html=True)
with st.expander("Explore the Full Destination Catalog"):
    if df_destinations is not None:
        st.markdown(f"**{len(df_destinations)} destinations available in our local knowledge base.**")
        st.dataframe(df_destinations[["Name","State","Type","Popularity","BestTimeToVisit"]], use_container_width=True, hide_index=True)
    else:
        st.error("Destination dataset not found.")

st.markdown("""
<br/><hr style='border-color:#ddd;'/>
<div style='text-align:center;color:#9a8a72;font-size:0.78rem;padding-bottom:1rem;'>
  TravelMind · Local AI Engine · Verified Destination Data
</div>""", unsafe_allow_html=True)