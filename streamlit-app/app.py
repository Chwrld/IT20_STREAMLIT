import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
import warnings
from pathlib import Path

from database import db_manager
from model_loader import load_artifacts, get_feature_info, get_accuracy, predict_from_dict, get_label_encoder, get_model, preprocess_dataframe
from schemas import PredictionInput, PredictionResult, FeatureInfo
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="TravelMind · Destination Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

def icon(name, size=18, color="currentColor", stroke_width: float = 2):
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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&family=Pacifico&family=Fredoka+One:wght@400;700&display=swap');
:root { --sand:#F5EDD8; --terra:#C97D4E; --deep:#2C2416; --sage:#6B8F71; --sky:#4A90A4; --card:#FFFDF6; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background-color:var(--sand); color:var(--deep); }
section[data-testid="stSidebar"] { background:var(--deep) !important; }
section[data-testid="stSidebar"] * { color:var(--sand) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label p {
    color:var(--terra) !important; font-weight:500; font-size:0.85rem; letter-spacing:0.05em; text-transform:uppercase;
}
section[data-testid="stSidebar"] .stSelectbox > div > div { background:var(--sand) !important; border:1px solid var(--terra) !important; color:var(--deep) !important; border-radius:6px !important; caret-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div > div { color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div:focus { background:var(--sand) !important; color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div { background:var(--sand) !important; color:var(--deep) !important; border-radius:6px !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] input { background:var(--sand) !important; color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] { background:var(--sand) !important; border:1px solid var(--terra) !important; border-radius:6px !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] div { color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] [role="option"] { color:var(--deep) !important; background:var(--sand) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] [role="option"]:hover { background:#e8dfc8 !important; color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] [data-baseweb="popover"] [role="option"][aria-selected="true"] { background:#e8dfc8 !important; color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] .stTextInput > div > input,
section[data-testid="stSidebar"] .stTextInput > div > div > input,
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] input[type="number"],
section[data-testid="stSidebar"] input[type="text"] { background:var(--sand) !important; border:1px solid var(--terra) !important; color:var(--deep) !important; border-radius:6px !important; caret-color:var(--deep) !important; -webkit-text-fill-color:var(--deep) !important; }
section[data-testid="stSidebar"] input::placeholder { color:#9a8a72 !important; opacity:1 !important; -webkit-text-fill-color:#9a8a72 !important; }
section[data-testid="stSidebar"] .stNumberInput button { background:var(--sand) !important; border:1px solid var(--terra) !important; color:var(--deep) !important; }
section[data-testid="stSidebar"] .stNumberInput button:hover { background:var(--terra) !important; color:white !important; }
section[data-testid="stSidebar"] .stNumberInput button svg { stroke:var(--deep) !important; }
section[data-testid="stSidebar"] .stNumberInput button:hover svg { stroke:white !important; }
.hero-title { font-family:'Fredoka One', cursive; font-size:3rem; font-weight:700; color:var(--deep); line-height:1.1; margin-bottom:0; text-shadow: 2px 2px 4px rgba(201,125,78,0.3); }
.hero-sub { font-size:1rem; color:#7a6a52; margin-top:0.25rem; margin-bottom:2rem; letter-spacing:0.04em; }
.pred-box { background:linear-gradient(135deg,var(--terra) 0%,#A0522D 100%); color:white; border-radius:20px; padding:2.5rem; text-align:center; box-shadow:0 8px 32px rgba(201,125,78,0.35); min-height:300px; display:flex; flex-direction:column; justify-content:center; background-size:cover; background-position:center; background-attachment:fixed; transition:all 0.3s ease; }
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
/* Expand sidebar width */
[data-testid="stSidebar"] { width: 570px !important; min-width: 550px !important; }
[data-testid="stSidebarContent"] { padding: 2rem 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Database Initialization ──
db_manager.init_db()

# ── Load trained model and artifacts ──
load_artifacts()
feature_info_data = get_feature_info() or {}
feature_info = FeatureInfo(feature_info_data)
accuracy = get_accuracy()
le = get_label_encoder()

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
    
    # Use the model_loader's predict function
    return predict_from_dict(input_dict)


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

    plan_name = st.text_input("Trip Plan Name", placeholder="e.g. Summer Vacation", value="")
    age_raw = st.text_input("Current Age", placeholder="e.g. 30", value="")
    age = int(age_raw) if age_raw.strip().isdigit() and 18 <= int(age_raw) <= 100 else None
    
    gender_options = ["🔹 Select Gender"] + feature_info.categorical_values["Gender"] + ["Other"]
    gender = st.selectbox("Gender Identity", gender_options, index=0)
    gender = None if gender == "🔹 Select Gender" else gender
    
    col_a, col_c = st.columns(2)
    with col_a:
        num_adults = st.number_input("Adults", 1, 10, 2, step=1)
    with col_c:
        num_children = st.number_input("Children", 0, 10, 0, step=1)

    st.markdown(f"""
    <hr style='border-color:#3D3020;margin:1.2rem 0;'/>
    <div class="sidebar-section-label">{icon("plane",size=14,color="#C97D4E")} Trip Details</div>
    """, unsafe_allow_html=True)

    budget_options = ["🔹 Select Budget"] + sorted(feature_info.categorical_values["Budget"])
    budget = st.selectbox("Travel Budget", budget_options, index=0)
    budget = None if budget == "🔹 Select Budget" else budget
    
    months = ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"]
    selected_month = st.selectbox("Travel Month", ["🔹 Select Month"] + months, index=0)
    selected_month = None if selected_month == "🔹 Select Month" else selected_month
    travel_month = months.index(selected_month) + 1 if selected_month else None

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
    
    # Show validation status
    validation_status = []
    if not age:
        validation_status.append("❌ Age (18-100 required)")
    else:
        validation_status.append(f"✅ Age: {age}")
    
    if not gender:
        validation_status.append("❌ Gender required")
    else:
        validation_status.append(f"✅ Gender: {gender}")
    
    if not budget:
        validation_status.append("❌ Budget required")
    else:
        validation_status.append(f"✅ Budget: {budget}")
    
    if travel_month is None:
        validation_status.append("❌ Month required")
    else:
        validation_status.append(f"✅ Month: {selected_month}")
    
    form_complete = all([age, gender, budget, travel_month is not None])
    
    st.caption(" • ".join(validation_status))
    predict_btn = st.button("Predict Ideal Destination", type="primary", disabled=not form_complete)


# ── MAIN ──
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;">
  {icon("compass",size=36,color="#C97D4E",stroke_width=1.5)}
  <div class="hero-title">TravelMind AI</div>
</div>
""", unsafe_allow_html=True)

main_tabs = st.tabs(["Place Suggestions", "Import CSV", "History"])

with main_tabs[0]:
    st.markdown('<div class="hero-sub">Our AI analyzed 1,510 traveler history records during the training phase.</div>', unsafe_allow_html=True)

    colors = ["#C97D4E","#6B8F71","#4A90A4","#8B6BA8","#D4A853"]
    col_pred, col_probs = st.columns([1.2,1.8], gap="large")

    DEST_IMAGES = {
        "Agra":               "https://www.onthegotours.com/repository/The-Taj-Mahal--India-Tours--On-The-Go-Tours-298431462895266.jpg",
        "Amritsar":           "https://sacredsites.com/images/asia/india/punjab/Golden-Temple-2.webp",
        "Andaman & Nicobar":  "https://www.explorebees.com/uploads/blogs/HOW+TO+REACH+Andaman+and+Nicobar+Islands.jpg",
        "Auroville":          "https://files.auroville.org/auroville-org/c10eba4a-5a80-45e5-b1fe-c49caba208cc.jpg",
        "Bodh Gaya":          "https://www.tusktravel.com/blog/wp-content/uploads/2025/05/How-to-Reach-Bodh-Gaya.jpg",
        "Delhi":              "https://static.toiimg.com/photo/msid-88070906,width-96,height-65.cms",
        "Dharamshala":        "https://c.ndtvimg.com/gws/ms/top-places-to-visit-in-dharamshala/assets/11.jpeg?1765125887",
        "Goa":                "https://static.businessworld.in/Untitled%20design%20-%202024-12-31T052430.892_20241231105033_original_image_31.webp",
        "Gokarna":            "https://templeinkarnataka.com/wp-content/uploads/2024/08/Mahabaleshwara-Temple1.png",
        "Hampi":              "https://carams.in/wp-content/uploads/2018/11/Hampi.jpg",
        "Hyderabad":          "https://t4.ftcdn.net/jpg/13/77/26/63/360_F_1377266312_jRydmbVRledy8RPzhOpRtPCNwIl46lEI.jpg",
        "Jaipur":             "https://dynamic-media-cdn.tripadvisor.com/media/photo-o/0a/c4/72/f6/jal-mahal-jaipur-tour.jpg?w=900&h=500&s=1",
        "Jim Corbett":        "https://uttarakhandtourism.gov.in/assets/media/UTDB_media_1735984081Jungle_safari.jpg",
        "Kochi":              "https://www.india.com/wp-content/uploads/2024/08/Fort-Kochi.jpg",
        "Kolkata":            "https://s7ap1.scene7.com/is/image/incredibleindia/victoria-memorial-kolkata-west-bengal-hero?qlt=82&ts=1742156385257",
        "Leh Ladakh":         "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSleU3dRyNbNjGJP8ZXOG9LLMVnogpJD2KSg&s",
        "Lucknow":            "https://media-cdn.tripadvisor.com/media/attractions-splice-spp-674x446/08/18/bb/a8.jpg",
        "Manali":             "https://www.oyorooms.com/travel-guide/wp-content/uploads/2022/03/Budget-Friendly-ways-to-travel-and-stay-in-Manali.jpg",
        "Munnar":             "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFatIUfMK22cIuPTluGZoVGifiH1fY9es5EA&s",
        "Mysore":             "https://cdn.britannica.com/58/124658-050-28314DA4/Maharaja-Palace-Mysuru-Karnataka-India.jpg",
        "Pondicherry":        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Pondicherry-Rock_beach_aerial_view.jpg/1280px-Pondicherry-Rock_beach_aerial_view.jpg",
        "Rishikesh":          "https://s7ap1.scene7.com/is/image/incredibleindia/1-triveni-ghat-rishikesh-uttarakhand-2-city-hero?qlt=82&ts=1726646286991",
        "Tirupati":           "https://www.hotelierindia.com/cloud/2025/03/20/Kodandarama-Temple-Tirupati-A-Travelers-Guide-to-a-Sacred-Site-1.jpg",
        "Varanasi":           "https://res.cloudinary.com/odysseytraveller/image/fetch/f_auto,q_auto,dpr_auto,r_4,w_765,h_535.5,c_limit/https://cdn.odysseytraveller.com/app/uploads/2018/05/iStock-827065008.jpg",
        "Varkala":            "https://s3.india.com/wp-content/uploads/2025/06/8-Relaxing-Weekend-Escapes-From-Varkala-For-Peace-Seekers.jpg",
    }

    with col_pred:
        st.markdown('<div class="section-head">Top Recommendation</div>', unsafe_allow_html=True)

        if predict_btn:
            try:
                if not all([age, gender, budget, travel_month]):
                    st.error("Please fill in all required fields before predicting.")
                    st.stop()
                
                prob_dict = predict(age, gender, num_adults, num_children, budget, travel_month, selected_prefs)
                pred_name = max(prob_dict, key=lambda k: prob_dict[k])
                pred_pct  = prob_dict[pred_name] * 100

                dest_svg = icon("map-pin", size=56, color="white", stroke_width=1.2)
                bg_img = DEST_IMAGES.get(pred_name, "")
                bg_style = (
                    f"background:linear-gradient(rgba(0,0,0,0.45),rgba(0,0,0,0.55)), url('{bg_img}') center/cover no-repeat;"
                    if bg_img else
                    "background:linear-gradient(135deg,var(--terra) 0%,#A0522D 100%);"
                )

                st.markdown(f"""
                <div class="pred-box" style="{bg_style}">
                  <div class="dest-icon">{dest_svg}</div>
                  <div class="dest-name">{pred_name}</div>
                  <div class="conf-label">Predicted Destination Class</div>
                  <div class="conf-pct">{pred_pct:.1f}% Match Confidence</div>
                  <div style="margin-top:1rem; font-size:0.8rem; opacity:0.9;">
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
                    month=selected_month,
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
        st.markdown('<div class="section-head">Likely Travel Destinations</div>', unsafe_allow_html=True)

        if "prob_dict" in st.session_state:
            prob_dict  = st.session_state["prob_dict"]
            pred_name  = max(prob_dict, key=lambda k: prob_dict[k])
            prob_pairs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

            bars_html = "<div style='display:flex;flex-direction:column;gap:1.2rem;height:100%;'>"
            for i, (dname, prob) in enumerate(prob_pairs):
                pct  = prob * 100
                clr  = colors[i % len(colors)]
                pin  = icon("map-pin", size=16, color=clr)
                bold = "font-weight:700;" if dname == pred_name else ""
                bars_html += f"""
                <div style="flex:1;display:flex;flex-direction:column;justify-content:center;gap:0.5rem;">
                  <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div style="display:flex;align-items:center;gap:0.4rem;{bold}font-size:1rem;color:#2C2416;">{pin} {dname}</div>
                    <div style="font-size:1rem;font-weight:700;color:{clr};">{pct:.1f}%</div>
                  </div>
                  <div style="background:#e8dfc8;border-radius:100px;height:22px;overflow:hidden;">
                    <div style="width:{pct:.1f}%;height:100%;border-radius:100px;background:{clr};transition:width 0.6s ease;"></div>
                  </div>
                </div>"""
            bars_html += "</div>"
            st.markdown(bars_html, unsafe_allow_html=True)
            st.markdown(f"""
            <div style='margin-top:1.2rem;font-size:0.77rem;color:#9a8a72;border-top:1px solid #e8dfc8;padding-top:0.8rem;'>
              <span style='font-family:"Fredoka One", cursive; color:#C97D4E; font-size:0.85rem;'>✨ TravelMind's AI Explorer</span> | Accuracy: <b>77.0%</b><br/>
              Output: Probabilities across <b>25 trained destination classes</b>.
            </div>""", unsafe_allow_html=True)
        else:
            st.caption("Match scores across trained classes will be visualized here.")

with main_tabs[1]:
    st.markdown('<div class="section-head">Batch Processing</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file to process multiple profiles and add predictions to history.")
    
    # CSV Format Guide
    with st.expander("📋 CSV Format Guide"):
        st.markdown("""
        **Required Columns:** Your CSV must include these exact column names:
        
        | Column | Type | Description | Example Values |
        |--------|------|-------------|----------------|
        | `Name` | Optional | Traveler name for better history tracking | "John Smith" |
        | `Age` | Required | Age in years | 25, 30, 45 |
        | `NumberOfAdults` | Required | Number of adults in group | 1, 2, 3 |
        | `NumberOfChildren` | Required | Number of children in group | 0, 1, 2 |
        | `TravelMonth` | Required | Month number (1-12) | 1=Jan, 6=Jun, 12=Dec |
        | `Gender` | Required | Gender identity | Male, Female, Other |
        | `Budget` | Required | Travel budget level | Low, Medium, High |
        | `Pref_Relaxation` | Required | Relaxation preference (0=No, 1=Yes) | 0, 1 |
        | `Pref_Adventure` | Required | Adventure preference (0=No, 1=Yes) | 0, 1 |
        | `Pref_Culture` | Required | Culture preference (0=No, 1=Yes) | 0, 1 |
        | `Pref_Spiritual` | Required | Spiritual preference (0=No, 1=Yes) | 0, 1 |
        
        **Example CSV:**
        ```csv
        Name,Age,NumberOfAdults,NumberOfChildren,TravelMonth,Gender,Budget,Pref_Relaxation,Pref_Adventure,Pref_Culture,Pref_Spiritual
        John Smith,25,2,0,6,Male,Medium,1,0,1,0
        Sarah Johnson,30,1,1,8,Female,High,0,1,1,0
        ```
        """)
        
        # Download sample CSV
        sample_csv = """Name,Age,NumberOfAdults,NumberOfChildren,TravelMonth,Gender,Budget,Pref_Relaxation,Pref_Adventure,Pref_Culture,Pref_Spiritual
John Smith,25,2,0,6,Male,Medium,1,0,1,0
Sarah Johnson,30,1,1,8,Female,High,0,1,1,0
Michael Brown,45,2,2,12,Male,Low,1,1,0,0
Emily Davis,35,3,0,4,Female,Medium,0,0,1,1
David Wilson,28,1,0,9,Male,High,1,1,0,1"""
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="sample_batch_data.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload Profile Data", type=["csv"])
    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.dataframe(batch_df.head(), width='stretch')
            
            # The pipeline handles scaling and encoding internally for these 10 features
            required = ["Age", "NumberOfAdults", "NumberOfChildren", "TravelMonth", "Gender", "Budget", 
                        "Pref_Relaxation", "Pref_Adventure", "Pref_Culture", "Pref_Spiritual"]
            missing = [c for c in required if c not in batch_df.columns]
            
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
                st.info("Check the CSV Format Guide above for the correct column names and format.")
            elif st.button("🚀 Process & Save to History"):
                with st.spinner("Processing batch and saving to history..."):
                    saved_count = 0
                    months_list = ["January", "February", "March", "April", "May", "June", 
                                 "July", "August", "September", "October", "November", "December"]
                    
                    for idx, row in batch_df.iterrows():
                        row_num = int(idx)  # type: ignore[arg-type]
                        try:
                            # Create input dict for prediction
                            input_dict = {
                                "Age": int(row["Age"]),
                                "NumberOfAdults": int(row["NumberOfAdults"]),
                                "NumberOfChildren": int(row["NumberOfChildren"]),
                                "TravelMonth": int(row["TravelMonth"]),
                                "Gender": row["Gender"],
                                "Budget": row["Budget"],
                                "Pref_Relaxation": int(row["Pref_Relaxation"]),
                                "Pref_Adventure": int(row["Pref_Adventure"]),
                                "Pref_Culture": int(row["Pref_Culture"]),
                                "Pref_Spiritual": int(row["Pref_Spiritual"])
                            }
                            
                            # Get prediction probabilities
                            prob_dict = predict_from_dict(input_dict)
                            pred_name = max(prob_dict, key=lambda k: prob_dict[k])
                            pred_pct = prob_dict[pred_name] * 100
                            
                            # Convert preferences back to list format
                            prefs_list = []
                            if row["Pref_Relaxation"]: prefs_list.append("Relaxation")
                            if row["Pref_Adventure"]: prefs_list.append("Adventure")
                            if row["Pref_Culture"]: prefs_list.append("Heritage & Culture")
                            if row["Pref_Spiritual"]: prefs_list.append("Spiritual")
                            
                            # Determine plan name - use Name column if available, otherwise use batch import
                            plan_name = row.get("Name", f"Batch Import {row_num + 1}")
                            if pd.isna(plan_name) or plan_name == "":
                                plan_name = f"Batch Import {row_num + 1}"
                            
                            # Save to database
                            db_manager.save_prediction(
                                plan_name=plan_name,
                                recommendation=pred_name,
                                match_confidence=pred_pct,
                                age=int(row["Age"]),
                                gender=row["Gender"],
                                adults=int(row["NumberOfAdults"]),
                                children=int(row["NumberOfChildren"]),
                                budget=row["Budget"],
                                month=months_list[int(row["TravelMonth"])-1],
                                prefs=prefs_list
                            )
                            saved_count += 1
                            
                        except Exception as row_error:
                            st.warning(f"Error processing row {row_num + 1}: {row_error}")
                            continue
                    
                    st.success(f"Successfully processed and saved {saved_count} predictions to history!")
                    st.info("Check the 'History' tab to view all saved predictions.")
                    
        except Exception as e:
            st.error(f"Batch processing error: {e}")

with main_tabs[2]:
    st.markdown('<div class="section-head">Trained Prediction Log</div>', unsafe_allow_html=True)
    history = db_manager.get_history()
    if history:
        st.dataframe(pd.DataFrame(history), width='stretch', hide_index=True)
        if st.button("Clear Prediction History"):
            db_manager.clear_history()
            st.rerun()
    else:
        st.info("No prediction history recorded in the database.")

# ── DATASET EXPLORER ──
st.markdown("<br/>", unsafe_allow_html=True)
with st.expander("Explore Trained Model Artifacts"):
    dest_data = {
        "Name": ["Agra","Amritsar","Andaman & Nicobar","Auroville","Bodh Gaya","Delhi","Dharamshala",
                 "Goa","Gokarna","Hampi","Hyderabad","Jaipur","Jim Corbett","Kochi","Kolkata",
                 "Leh Ladakh","Lucknow","Manali","Munnar","Mysore","Pondicherry","Rishikesh",
                 "Tirupati","Varanasi","Varkala"],
        "State": ["Uttar Pradesh","Punjab","Andaman & Nicobar Islands","Tamil Nadu","Bihar",
                  "Delhi","Himachal Pradesh","Goa","Karnataka","Karnataka","Telangana","Rajasthan",
                  "Uttarakhand","Kerala","West Bengal","Jammu & Kashmir","Uttar Pradesh",
                  "Himachal Pradesh","Kerala","Karnataka","Puducherry","Uttarakhand",
                  "Andhra Pradesh","Uttar Pradesh","Kerala"],
        "Type": ["Historical","Spiritual","Beach","Nature","Spiritual","City","Nature",
                 "Beach","Beach","Historical","City","Historical","Nature","City","City",
                 "Adventure","Historical","Adventure","Nature","Historical","Beach","Adventure",
                 "Spiritual","Spiritual","Beach"],
        "Popularity": [9.1,8.7,8.9,7.8,8.2,8.8,8.4,9.0,8.3,8.6,8.5,9.2,8.1,8.7,8.3,
                       8.9,7.9,8.8,8.6,8.7,8.4,8.9,8.0,9.1,8.2],
        "BestTimeToVisit": ["Oct-Mar","Oct-Mar","Nov-May","Nov-Feb","Oct-Mar","Oct-Mar","Mar-Jun",
                            "Nov-Mar","Oct-Mar","Oct-Feb","Oct-Mar","Oct-Mar","Nov-Jun","Sep-Mar",
                            "Oct-Mar","Apr-Jun","Oct-Mar","Mar-Jun","Sep-Mar","Oct-Mar","Oct-Mar",
                            "Sep-Jun","Oct-Mar","Oct-Mar","Sep-Mar"],
    }
    df_artifacts = pd.DataFrame(dest_data)
    st.markdown(f"**{len(df_artifacts)} destination classes · learned by the trained model**")
    tab1, tab2 = st.tabs(["Destinations", "Summary Stats"])
    with tab1:
        st.dataframe(df_artifacts, width='stretch', hide_index=True)
    with tab2:
        st.markdown("### Destination Match Analysis")
        try:
            # Create bar chart showing destination names and their match scores (popularity)
            fig = go.Figure(data=[
                go.Bar(
                    x=df_artifacts["Name"],
                    y=df_artifacts["Popularity"],
                    marker_color='#C97D4E',
                    text=df_artifacts["Popularity"],
                    textposition='auto',
                    textfont=dict(size=10, color='white'),
                    hovertemplate='<b>%{x}</b><br>Match Score: %{y:.1f}<extra></extra>'
                )
            ])
            fig.update_layout(
                title=dict(
                    text="Destination Match Scores",
                    font=dict(size=16, color="#2C2416", family="DM Sans"),
                    x=0.5
                ),
                xaxis_title="Destinations",
                yaxis_title="Match Score",
                plot_bgcolor="#FFFDF6",
                paper_bgcolor="#F5EDD8",
                xaxis=dict(
                    tickfont=dict(size=10, color="#2C2416"),
                    tickangle=45,
                    gridcolor="#e8dfc8"
                ),
                yaxis=dict(
                    tickfont=dict(size=12, color="#2C2416"),
                    gridcolor="#e8dfc8",
                    zeroline=False
                ),
                height=400,
                margin=dict(l=20, r=20, t=50, b=80)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Add summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Destinations", len(df_artifacts))
            with col2:
                st.metric("Highest Match Score", f"{df_artifacts['Popularity'].max():.1f}")
            with col3:
                st.metric("Average Match Score", f"{df_artifacts['Popularity'].mean():.1f}")
                
        except Exception as e:
            st.error(f"Error generating match analysis: {e}")
            # Fallback to basic table
            st.dataframe(
                df_artifacts[["Name", "Popularity"]].sort_values("Popularity", ascending=False),
                width='stretch'
            )

st.markdown("---")
st.markdown(
    f'<p style="text-align: center; color: grey; font-size: 0.8rem;">'
    f'TravelMind AI · Built with Streamlit & Trained joblib artifacts · Matching across {len(le.classes_) if le is not None else 25} Classes'
    '</p>', 
    unsafe_allow_html=True
)