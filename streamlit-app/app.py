import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from database import db_manager

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TravelMind · Destination Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stCheckbox label p {
    color:var(--terra) !important; font-weight:500; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase;
}
section[data-testid="stSidebar"] .stSelectbox > div > div { background:#3D3020 !important; border:1px solid var(--terra) !important; color:var(--sand) !important; }
section[data-testid="stSidebar"] .stTextInput > div > input { background:#3D3020 !important; border:1px solid var(--terra) !important; color:var(--sand) !important; border-radius:6px !important; }
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
.accuracy-badge { display:inline-block; background:rgba(107,143,113,0.15); border:1px solid var(--sage); border-radius:20px; padding:0.25rem 0.9rem; font-size:0.78rem; color:var(--sage); font-weight:600; margin-top:0.8rem; }
.stDataFrame { border-radius:12px; overflow:hidden; }
.stButton > button { background:var(--terra) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:600 !important; letter-spacing:0.04em !important; padding:0.65rem 2rem !important; font-size:0.95rem !important; transition:opacity 0.2s !important; width:100% !important; }
.stButton > button:hover { opacity:0.85 !important; }
</style>
""", unsafe_allow_html=True)

# ── Database Initialization ──
db_manager.init_db()

# ── Load trained model and artifacts from notebook ──
@st.cache_resource
def load_trained_model():
    # Using Path(__file__) logic from professor's design
    ROOT = Path(__file__).resolve().parents[1]
    base_path = ROOT / "models"
    
    # Loading strictly the 4 artifacts derived from the joblib training process
    model = joblib.load(base_path / "best_model.joblib")
    preprocessor = joblib.load(base_path / "preprocessor.joblib")
    le = joblib.load(base_path / "label_encoder.joblib")
    X_columns = joblib.load(base_path / "X_columns.joblib")
    
    # Notebook confirmed 20.1% accuracy for the best model on the test set
    return model, le, preprocessor, X_columns, 20.1

model, le, preprocessor, X_columns, accuracy = load_trained_model()

# ── Prediction Logic ──
def predict(age, gender, num_adults, num_children, budget, travel_month, prefs):
    # Mapping preferences to binary features as expected by the trained preprocessor
    input_dict = {
        "Age": age,
        "NumberOfAdults": num_adults,
        "NumberOfChildren": num_children,
        "TravelMonth": travel_month,
        "Gender": gender,
        "Budget": budget,
        "Pref_Relaxation": 1 if "Relaxation" in prefs else 0,
        "Pref_Adventure": 1 if "Adventure" in prefs else 0,
        "Pref_Culture": 1 if "Heritage & Culture" in prefs else 0,
        "Pref_Spiritual": 1 if "Spiritual" in prefs else 0
    }
    input_data = pd.DataFrame([input_dict])
    
    # model is a Pipeline object containing both the preprocessor and the classifier
    probs = model.predict_proba(input_data)[0]
    return dict(zip(le.classes_, probs))


# ── SIDEBAR ──
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:1rem 0 0.5rem;'>
      <div style='display:flex;justify-content:center;'>{icon("compass",size=40,color="#C97D4E",stroke_width=1.5)}</div>
      <div style='font-family:"Playfair Display",serif;font-size:1.3rem;font-weight:700;color:#F5EDD8;margin-top:0.5rem;'>TravelMind</div>
      <div style='font-size:0.72rem;letter-spacing:0.12em;color:#C97D4E;text-transform:uppercase;'>Trained AI Predictor</div>
    </div>
    <hr style='border-color:#3D3020;margin:0.8rem 0 1.2rem;'/>
    <div class="sidebar-section-label">{icon("user",size=14,color="#C97D4E")} User Profile</div>
    """, unsafe_allow_html=True)

    plan_name = st.text_input("Trip Plan Name", value="My Awesome Trip", placeholder="e.g. Summer Vacation")
    age = st.slider("Current Age", 18, 90, 30)
    gender = st.selectbox("Gender Identity", ["Male", "Female", "Other"])
    
    col_a, col_c = st.columns(2)
    with col_a:
        num_adults = st.number_input("Adults", 1, 10, 2)
    with col_c:
        num_children = st.number_input("Children", 0, 10, 0)

    st.markdown(f"""
    <hr style='border-color:#3D3020;margin:1.2rem 0;'/>
    <div class="sidebar-section-label">{icon("plane",size=14,color="#C97D4E")} Trip Details</div>
    """, unsafe_allow_html=True)

    budget = st.selectbox("Travel Budget", ["Low", "Medium", "High"], index=1)
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Travel Month", months, index=10) # Default Nov
    travel_month = months.index(selected_month) + 1

    st.markdown('<div class="sidebar-section-label">Travel Interests</div>', unsafe_allow_html=True)
    
    pref_relax = st.checkbox("Relaxation", value=True)
    pref_adv = st.checkbox("Adventure", value=False)
    pref_culture = st.checkbox("Heritage & Culture", value=True)
    pref_spirit = st.checkbox("Spiritual", value=False)
    
    selected_prefs = []
    if pref_relax: selected_prefs.append("Relaxation")
    if pref_adv: selected_prefs.append("Adventure")
    if pref_culture: selected_prefs.append("Heritage & Culture")
    if pref_spirit: selected_prefs.append("Spiritual")

    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("✨ Predict Ideal Destination", type="primary")


# ── MAIN ──
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;">
  {icon("compass",size=36,color="#C97D4E",stroke_width=1.5)}
  <div class="hero-title">TravelMind AI</div>
</div>
""", unsafe_allow_html=True)

main_tabs = st.tabs(["✨ Discovery Engine", "📂 Pipeline Processing", "📜 Trained History"])

with main_tabs[0]:
    st.markdown('<div class="hero-sub">Our AI analyzed 1,510 traveler history records during the training phase.</div>', unsafe_allow_html=True)

    colors = ["#C97D4E","#6B8F71","#4A90A4","#8B6BA8","#D4A853"]
    col_pred, col_probs = st.columns([1.2,1.8], gap="large")

    with col_pred:
        st.markdown('<div class="section-head">Top Recommendation</div>', unsafe_allow_html=True)

        if predict_btn:
            try:
                prob_dict = predict(age, gender, num_adults, num_children, budget, travel_month, selected_prefs)
                pred_name = max(prob_dict, key=prob_dict.get)
                pred_pct  = prob_dict[pred_name] * 100

                dest_svg = icon("map-pin", size=56, color="white", stroke_width=1.2)

                st.markdown(f"""
                <div class="pred-box">
                  <div class="dest-icon">{dest_svg}</div>
                  <div class="dest-name">{pred_name}</div>
                  <div class="conf-label">Predicted Destination Class</div>
                  <div class="conf-pct">{pred_pct:.1f}% Match Confidence</div>
                  <div style="margin-top:1rem; font-size:0.8rem; opacity:0.8;">
                    Result derived from trained label encoder classes.
                  </div>
                </div>""", unsafe_allow_html=True)

                st.session_state["prob_dict"] = prob_dict
                
                # Save to SQLite Database
                db_manager.save_prediction(
                    plan_name=plan_name,
                    recommendation=pred_name,
                    match_confidence=pred_pct,
                    age=age,
                    gender=gender,
                    adults=num_adults,
                    children=num_children,
                    budget=budget,
                    month=months[travel_month-1],
                    prefs=selected_prefs
                )
            except Exception as e:
                st.error(f"Execution Error: {e}")
            
        else:
            if "prob_dict" not in st.session_state:
                st.info("Input traveler profile details in the sidebar and click **Predict Ideal Destination**.")
            else:
                st.caption("Showing most recent prediction from joblib artifacts.")

    with col_probs:
        st.markdown('<div class="section-head">Multinomial Probability Distribution</div>', unsafe_allow_html=True)

        if "prob_dict" in st.session_state:
            prob_dict  = st.session_state["prob_dict"]
            pred_name  = max(prob_dict, key=prob_dict.get)
            prob_pairs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

            bars_html = ""
            for i, (dname, prob) in enumerate(prob_pairs):
                pct  = prob * 100
                clr  = colors[i % len(colors)]
                pin  = icon("map-pin", size=13, color=clr)
                bold = "font-weight:700;" if dname == pred_name else ""
                bars_html += f"""
                <div class="prob-row">
                  <div class="prob-label" style="display:flex;align-items:center;gap:0.3rem;{bold}">{pin} {dname}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{clr};"></div>
                  </div>
                  <div class="prob-pct" style="color:{clr};">{pct:.1f}%</div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:1rem;font-size:0.77rem;color:#9a8a72;border-top:1px solid #e8dfc8;padding-top:0.8rem;'>
              Model: <b>Multinomial Logistic Regression Pipeline</b> | Accuracy: <b>{accuracy:.1f}%</b><br/>
              Output: Probabilities across <b>{len(le.classes_)}</b> trained destination classes.
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("Match scores across trained classes will be visualized here.")

with main_tabs[1]:
    st.markdown('<div class="section-head">Batch Pipeline Analysis</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file to process multiple profiles through the trained feature pipeline.")
    
    uploaded_file = st.file_uploader("Upload Profile Data", type=["csv"])
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # The pipeline handles scaling and encoding internally for these 10 features
            required = ["Age", "NumberOfAdults", "NumberOfChildren", "TravelMonth", "Gender", "Budget", 
                        "Pref_Relaxation", "Pref_Adventure", "Pref_Culture", "Pref_Spiritual"]
            missing = [c for c in required if c not in batch_df.columns]
            
            if missing:
                st.error(f"Missing required metadata columns: {missing}")
            elif st.button("🚀 Process Pipeline"):
                with st.spinner("Executing model..."):
                    preds = model.predict(batch_df[required])
                    batch_df["Recommendation"] = le.inverse_transform(preds)
                    st.success(f"Classified {len(batch_df)} records successfully.")
                    st.dataframe(batch_df, use_container_width=True)
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Classified Data", data=csv, file_name="classified_travelers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Pipeline error: {e}")

with main_tabs[2]:
    st.markdown('<div class="section-head">Trained Prediction Log</div>', unsafe_allow_html=True)
    history = db_manager.get_history()
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
        if st.button("Clear Prediction History"):
            db_manager.clear_history()
            st.rerun()
    else:
        st.info("No prediction history recorded in the database.")

# ── ARTIFACT EXPLORER ──
st.markdown("<br/>", unsafe_allow_html=True)
with st.expander("Explore Trained Model Artifacts"):
    st.markdown(f"**This application is strictly aligned with the following trained joblib artifacts:**")
    st.code("- best_model.joblib (Pipeline)\n- preprocessor.joblib (ColumnTransformer)\n- label_encoder.joblib (25 classes)\n- X_columns.joblib (Input schema)")
    
    t1, t2 = st.tabs(["Target Classes", "Input Schema"])
    with t1:
        st.write("The following 25 destination classes were learned by the model:")
        st.write(", ".join(le.classes_))
    with t2:
        st.write("The model expects the following 10 raw features as input:")
        st.write(", ".join(["Age", "NumberOfAdults", "NumberOfChildren", "TravelMonth", "Gender", "Budget", "Pref_Relaxation", "Pref_Adventure", "Pref_Culture", "Pref_Spiritual"]))

st.markdown("---")
st.markdown(
    f'<p style="text-align: center; color: grey; font-size: 0.8rem;">'
    f'TravelMind AI · Built with Streamlit & Trained joblib artifacts · Matching across {len(le.classes_)} Classes'
    '</p>', 
    unsafe_allow_html=True
)