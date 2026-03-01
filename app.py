import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
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
section[data-testid="stSidebar"] .stTextInput label {
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


# ── Build & train model from synthetic user-destination data ──
@st.cache_resource
def train_model():
    np.random.seed(42)

    # Realistic traveler profiles per destination
    PROFILES = {
        "Taj Mahal":         {"age": (42,12), "adults": (2.5,1.0), "children": (1.2,1.0),
                              "gender": [0.45,0.45,0.10],
                              "dtype":  {"Historical":0.55,"City":0.25,"Nature":0.08,"Beach":0.07,"Adventure":0.05},
                              "season": {"Oct-Dec":0.40,"Jan-Mar":0.35,"Jul-Sep":0.15,"Apr-Jun":0.10}},
        "Goa Beaches":       {"age": (28, 8), "adults": (3.5,1.2), "children": (0.3,0.6),
                              "gender": [0.50,0.40,0.10],
                              "dtype":  {"Beach":0.60,"Nature":0.15,"City":0.12,"Adventure":0.08,"Historical":0.05},
                              "season": {"Jan-Mar":0.45,"Oct-Dec":0.35,"Jul-Sep":0.12,"Apr-Jun":0.08}},
        "Jaipur City":       {"age": (38,11), "adults": (3.0,1.1), "children": (1.0,1.0),
                              "gender": [0.42,0.48,0.10],
                              "dtype":  {"City":0.50,"Historical":0.30,"Nature":0.10,"Beach":0.06,"Adventure":0.04},
                              "season": {"Oct-Dec":0.42,"Jan-Mar":0.38,"Apr-Jun":0.12,"Jul-Sep":0.08}},
        "Kerala Backwaters": {"age": (44,10), "adults": (2.2,0.9), "children": (0.8,0.9),
                              "gender": [0.40,0.50,0.10],
                              "dtype":  {"Nature":0.55,"Beach":0.22,"City":0.10,"Historical":0.08,"Adventure":0.05},
                              "season": {"Jul-Sep":0.38,"Oct-Dec":0.32,"Jan-Mar":0.20,"Apr-Jun":0.10}},
        "Leh Ladakh":        {"age": (27, 7), "adults": (3.8,1.3), "children": (0.1,0.3),
                              "gender": [0.60,0.32,0.08],
                              "dtype":  {"Adventure":0.60,"Nature":0.28,"City":0.05,"Historical":0.04,"Beach":0.03},
                              "season": {"Apr-Jun":0.50,"Jul-Sep":0.38,"Jan-Mar":0.07,"Oct-Dec":0.05}},
    }

    rows = []
    for dest, p in PROFILES.items():
        for _ in range(400):
            age      = int(np.clip(np.random.normal(*p["age"]), 18, 75))
            gender   = np.random.choice(["Male","Female","Other"], p=p["gender"])
            adults   = max(1, int(np.round(np.random.normal(*p["adults"]))))
            children = max(0, int(np.round(np.random.normal(*p["children"]))))
            dtype    = np.random.choice(list(p["dtype"].keys()), p=list(p["dtype"].values()))
            season   = np.random.choice(list(p["season"].keys()), p=list(p["season"].values()))
            rows.append({"Age":age,"Gender":gender,"NumberOfAdults":adults,
                         "NumberOfChildren":children,"DestType":dtype,
                         "TravelSeason":season,"Destination":dest})

    df       = pd.DataFrame(rows)
    le       = LabelEncoder()
    y        = le.fit_transform(df["Destination"])
    cat_cols = ["Gender","DestType","TravelSeason"]
    df_enc   = pd.get_dummies(df[cat_cols])
    enc_cols = df_enc.columns.tolist()
    num_cols = ["Age","NumberOfAdults","NumberOfChildren"]
    scaler   = StandardScaler()
    X_num    = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
    X        = np.hstack([df_enc.values, X_num.values])

    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X, y)

    # Compute CV accuracy for display
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy").mean()

    return model, le, enc_cols, num_cols, scaler, round(acc * 100, 1)


model, le, enc_cols, num_cols, scaler, cv_accuracy = train_model()


def predict(age, gender, num_adults, num_children, dest_type, travel_season):
    cat_df  = pd.DataFrame([{"Gender":gender,"DestType":dest_type,"TravelSeason":travel_season}])
    cat_enc = pd.get_dummies(cat_df)
    for col in enc_cols:
        if col not in cat_enc.columns: cat_enc[col] = 0
    cat_enc = cat_enc[enc_cols]
    num_arr = scaler.transform([[age, num_adults, num_children]])
    num_df  = pd.DataFrame(num_arr, columns=num_cols)
    X       = np.hstack([cat_enc.values, num_df.values])
    probs   = model.predict_proba(X)[0]
    return dict(zip(le.classes_, probs))


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

    age_raw      = st.text_input("Age", value="28", placeholder="e.g. 25")
    age          = int(age_raw) if age_raw.strip().isdigit() else 28
    gender       = st.selectbox("Gender", ["Male","Female","Other"])
    adults_raw   = st.text_input("Number of Adults", value="2", placeholder="e.g. 2")
    num_adults   = max(1, int(adults_raw) if adults_raw.strip().isdigit() else 2)
    children_raw = st.text_input("Number of Children", value="0", placeholder="e.g. 0")
    num_children = max(0, int(children_raw) if children_raw.strip().isdigit() else 0)

    st.markdown(f"""
    <hr style='border-color:#3D3020;margin:1.2rem 0;'/>
    <div class="sidebar-section-label">{icon("plane",size=14,color="#C97D4E")} Your Trip</div>
    """, unsafe_allow_html=True)

    dest_type     = st.selectbox("What type of destination appeals to you?",
                                 ["Beach","Historical","City","Nature","Adventure"])
    travel_season = st.selectbox("When do you plan to travel?",
                                 ["Jan-Mar","Apr-Jun","Jul-Sep","Oct-Dec"])


    st.markdown("<br/>", unsafe_allow_html=True)
    predict_btn = st.button("Find My Destination")


# ── MAIN ──
st.markdown(f"""
<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.25rem;">
  {icon("compass",size=36,color="#C97D4E",stroke_width=1.5)}
  <div class="hero-title">Where Should You Travel Next?</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Answer a few questions and let AI find your perfect Indian destination.</div>', unsafe_allow_html=True)

dest_icon_map = {
    "Taj Mahal":         ("landmark","Uttar Pradesh","Historical","Nov–Feb"),
    "Goa Beaches":       ("waves",   "Goa",          "Beach",     "Nov–Mar"),
    "Jaipur City":       ("city",    "Rajasthan",    "City",      "Oct–Mar"),
    "Kerala Backwaters": ("leaf",    "Kerala",       "Nature",    "Sep–Mar"),
    "Leh Ladakh":        ("mountain","J&K",          "Adventure", "Apr–Jun"),
}
colors = ["#C97D4E","#6B8F71","#4A90A4","#8B6BA8","#D4A853"]

col_pred, col_probs = st.columns([1.2,1.8], gap="large")

with col_pred:
    st.markdown('<div class="section-head">Your Perfect Match</div>', unsafe_allow_html=True)

    if predict_btn:
        prob_dict = predict(age, gender, num_adults, num_children, dest_type, travel_season)
        pred_name = max(prob_dict, key=prob_dict.get)
        pred_pct  = prob_dict[pred_name] * 100

        icon_name, region, dtype, season = dest_icon_map.get(pred_name, ("map-pin","India","—","—"))
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
            <span class="dest-detail-item">{pin_svg} {region}</span>
            &nbsp;·&nbsp;
            <span class="dest-detail-item">{tag_svg} {dtype}</span>
            &nbsp;·&nbsp;
            <span class="dest-detail-item">{cal_svg} Best: {season}</span>
          </div>
        </div>""", unsafe_allow_html=True)

        st.session_state["prob_dict"] = prob_dict
    else:
        st.info("Answer the questions in the sidebar and click **Find My Destination**.")

with col_probs:
    st.markdown('<div class="section-head">How Well Each Destination Fits You</div>', unsafe_allow_html=True)

    if "prob_dict" in st.session_state:
        prob_dict  = st.session_state["prob_dict"]
        pred_name  = max(prob_dict, key=prob_dict.get)
        prob_pairs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

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
        st.markdown("""
        <div style='margin-top:1rem;font-size:0.77rem;color:#9a8a72;border-top:1px solid #e8dfc8;padding-top:0.8rem;'>
          Probabilities are generated by a Logistic Regression model trained on 2,000 synthetic 
          traveler profiles built from realistic destination archetypes.
        </div>""", unsafe_allow_html=True)
    else:
        st.caption("Match scores will appear here after prediction.")

# ── DATASET EXPLORER ──
@st.cache_data
def load_raw_dataset():
    """Try CSV first, then Excel, then embedded fallback."""
    path = "Expanded_Destinations.csv"
    # 1. Plain CSV (most common on local machines)
    try:
        df = pd.read_csv(path)
        if "Name" in df.columns:
            return df
    except Exception:
        pass
    # 2. Excel / xlsx disguised as .csv
    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        pass
    # 3. xlrd engine
    try:
        return pd.read_excel(path, engine="xlrd")
    except Exception:
        pass
    # 4. Embedded fallback
    return pd.DataFrame({
        "Name":            ["Taj Mahal","Goa Beaches","Jaipur City","Kerala Backwaters","Leh Ladakh"],
        "State":           ["Uttar Pradesh","Goa","Rajasthan","Kerala","Jammu and Kashmir"],
        "Type":            ["Historical","Beach","City","Nature","Adventure"],
        "Popularity":      [8.69, 8.61, 9.23, 7.98, 8.40],
        "BestTimeToVisit": ["Nov-Feb","Nov-Mar","Oct-Mar","Sep-Mar","Apr-Jun"],
    })

st.markdown("<br/>", unsafe_allow_html=True)
with st.expander("Explore the Destination Dataset"):
    df_raw = load_raw_dataset()
    st.markdown(f"**{len(df_raw)} records · {df_raw['Name'].nunique()} unique destinations**")
    tab1, tab2 = st.tabs(["Destinations", "Summary Stats"])
    with tab1:
        st.dataframe(df_raw[["Name","State","Type","Popularity","BestTimeToVisit"]].drop_duplicates("Name"), use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(df_raw.groupby("Name")["Popularity"].agg(["mean","min","max","count"]).rename(columns={"mean":"Avg Popularity","min":"Min","max":"Max","count":"Records"}).round(3), use_container_width=True)

st.markdown("""
<br/><hr style='border-color:#ddd;'/>
<div style='text-align:center;color:#9a8a72;font-size:0.78rem;padding-bottom:1rem;'>
  TravelMind · Logistic Regression · Trained on 2,000 Traveler Profiles ·
</div>""", unsafe_allow_html=True)