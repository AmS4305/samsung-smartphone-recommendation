"""
Samsung Smartphone Recommendation System - Streamlit App
Replicates the Jupyter notebook ML pipeline in a polished UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Samsung SmartPick",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Hero Header ── */
    .hero {
        background: linear-gradient(135deg, #1428A0 0%, #0B0E3F 60%, #000 100%);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(20,40,160,0.4) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        color: #fff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        color: rgba(255,255,255,0.7);
        font-size: 1.05rem;
        margin: 0;
        font-weight: 300;
    }
    .hero .accent {
        color: #60A5FA;
    }
    
    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(96,165,250,0.15);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #60A5FA;
        margin: 0;
    }
    .metric-card .label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }
    
    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 1rem 0;
    }
    .section-header .icon {
        font-size: 1.5rem;
    }
    .section-header h2 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
    }
    
    /* ── Price Result ── */
    .price-result {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(96,165,250,0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .price-result .price-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0.5rem;
    }
    .price-result .price-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .price-result .price-note {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.4);
        margin-top: 0.5rem;
    }
    
    /* ── Recommendation Card ── */
    .rec-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(96,165,250,0.12);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        border-color: rgba(96,165,250,0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(96,165,250,0.08);
    }
    .rec-card .rank {
        background: linear-gradient(135deg, #1428A0, #60A5FA);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .rec-card .rec-info {
        flex: 1;
        min-width: 0;
    }
    .rec-card .rec-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .rec-card .rec-score {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.45);
        margin-top: 0.15rem;
    }
    .rec-card .rec-price {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60A5FA;
        flex-shrink: 0;
    }
    .rec-card .rec-img {
        width: 56px;
        height: 56px;
        object-fit: contain;
        border-radius: 10px;
        background: white;
        padding: 4px;
        flex-shrink: 0;
    }
    
    /* ── Similarity Bar ── */
    .sim-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 3px;
        margin-top: 0.4rem;
        overflow: hidden;
    }
    .sim-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: linear-gradient(90deg, #1428A0, #60A5FA);
    }
    
    /* ── Spec Tag ── */
    .spec-tags {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .spec-tag {
        background: rgba(96,165,250,0.1);
        color: #93c5fd;
        font-size: 0.7rem;
        padding: 0.2rem 0.55rem;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* ── Selected Phone Card ── */
    .selected-phone {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid rgba(96,165,250,0.25);
        border-radius: 20px;
        padding: 1.8rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .selected-phone img {
        width: 120px;
        height: 120px;
        object-fit: contain;
        margin-bottom: 1rem;
        background: white;
        border-radius: 14px;
        padding: 8px;
    }
    .selected-phone .phone-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .selected-phone .phone-price {
        font-size: 1.3rem;
        font-weight: 700;
        color: #60A5FA;
        margin-top: 0.3rem;
    }
    
    /* ── Model Stats ── */
    .model-stats {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        border: 1px solid rgba(52,211,153,0.2);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
    .model-stats .stat-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(52,211,153,0.7);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .model-stats .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .model-stats .stat-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.6);
    }
    .model-stats .stat-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: #34d399;
    }
    
    /* ── Streamlit overrides ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0E3F 0%, #0f172a 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown h3 {
        color: #60A5FA;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1428A0, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(20,40,160,0.3) !important;
    }
    .stSelectbox label, .stSlider label {
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────
# Data Loading & Pipeline (cached)
# ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_process_data():
    """Replicate the full notebook data pipeline."""
    # ── Load datasets ──
    df1 = pd.read_csv("project1.csv")
    df2 = pd.read_csv("project2.csv")

    # ── Clean df1 ──
    df1["processor"] = df1["processor"].str.replace("not mentioned", "", regex=False)

    # ── Clean df2 ──
    df2["price"] = (
        df2["price"]
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(int)
    )
    df2["internal_storage"] = (
        df2["storage_ram"]
        .str.extract(r"Internal Storage(\d+) GB", expand=False)
        .astype(float)
    )
    df2["ram"] = (
        df2["storage_ram"].str.extract(r"RAM(\d+) GB", expand=False).astype(float)
    )
    df2["os"] = (
        df2["os_processor"]
        .str.extract(r"Operating System(.+?)Processor", expand=False)
        .str.strip()
    )
    df2["primary_camera"] = (
        df2["camera"].str.extract(r"Primary Camera(\d+)MP", expand=False).astype(float)
    )
    df2["display_size"] = (
        df2["display"]
        .str.extract(r"Display Size(.+?) cm", expand=False)
        .str.replace("(", "", regex=False)
        .astype(float)
    )
    df2["network_type"] = (
        df2["network"]
        .str.extract(r"Network Type(.+?)Supported", expand=False)
        .str.strip()
    )
    df2["supported_networks"] = (
        df2["network"].str.extract(r"Supported Networks(.+)", expand=False).str.strip()
    )
    df2["battery_capacity"] = (
        df2["battery"]
        .str.extract(r"Battery Capacity(\d+) mAh", expand=False)
        .fillna(-1)
        .astype(int)
    )

    df2.drop(
        ["storage_ram", "os_processor", "camera", "display", "network", "battery"],
        axis=1,
        inplace=True,
    )

    # ── Impute missing values (random sampling) ──
    np.random.seed(42)
    for col in ["network_type", "supported_networks", "ratings", "os"]:
        mask = df2[col].isnull()
        if mask.sum() > 0:
            df2.loc[mask, col] = (
                df2[col].dropna().sample(mask.sum(), random_state=42).values
            )

    # ── KNN impute numeric columns ──
    numeric_cols = [
        "ratings",
        "price",
        "internal_storage",
        "ram",
        "primary_camera",
        "display_size",
        "battery_capacity",
    ]
    knn = KNNImputer(n_neighbors=2)
    imputed = knn.fit_transform(df2[numeric_cols])
    X2 = pd.DataFrame(imputed, columns=numeric_cols)

    cat_cols = df2[["name", "imgURL", "os", "network_type", "supported_networks"]]
    df3 = pd.concat([cat_cols, X2], axis=1)

    # ── One-hot encode ──
    counts_net = df3["network_type"].value_counts()
    repl_net = counts_net[counts_net <= 100].index
    Y = pd.get_dummies(
        df3["network_type"].replace(repl_net, "uncommon"), drop_first=True
    ).astype(int)

    counts_os = df3["os"].value_counts()
    repl_os = counts_os[counts_os <= 50].index
    Z = pd.get_dummies(df3["os"].replace(repl_os, "others"), drop_first=True).astype(
        int
    )

    D = pd.concat([Y, Z], axis=1)
    df4 = pd.concat([df3, D], axis=1)
    df4.drop(
        ["imgURL", "os", "network_type", "supported_networks"], axis=1, inplace=True
    )

    # ── Clustering ──
    feature_cols = [c for c in df4.columns if c not in ["name"]]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df4[feature_cols])
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(data_scaled)
    df4["cluster"] = kmeans.labels_

    # ── Rename network col ──
    if "5G, 4G, 3G, 2G" in df4.columns:
        df4.rename(columns={"5G, 4G, 3G, 2G": "Network"}, inplace=True)

    # ── Build image/price lookup from original df2 ──
    img_lookup = (
        pd.read_csv("project2.csv")[["name", "imgURL"]]
        .drop_duplicates(subset="name")
        .set_index("name")["imgURL"]
        .to_dict()
    )
    img_lookup1 = (
        pd.read_csv("project1.csv")[["name", "imgURL"]]
        .drop_duplicates(subset="name")
        .set_index("name")["imgURL"]
        .to_dict()
    )
    img_lookup.update(img_lookup1)

    return df4, img_lookup


@st.cache_resource
def train_model(df4):
    """Train the GradientBoostingRegressor for price prediction."""
    model_features = [
        "internal_storage",
        "ram",
        "primary_camera",
        "display_size",
        "cluster",
    ]
    # Add optional columns if they exist
    for col in ["Network", "Android 13"]:
        if col in df4.columns:
            model_features.append(col)

    data_selected = df4[model_features + ["price"]].dropna(subset=["price"])
    X = data_selected.drop(columns=["price"])
    y = data_selected["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = GradientBoostingRegressor(random_state=0)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    n = len(X_test)
    p = X_test.shape[1]
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return (
        model,
        model_features,
        {
            "r2": r2,
            "adj_r2": adj_r2,
            "mae": mae,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
    )


def get_recommendations(prod_name, data, features, top_n=5):
    """Get top N similar phones using cosine similarity."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[features])
    sim_matrix = cosine_similarity(scaled)

    idx_matches = data.index[data["name"] == prod_name]
    if len(idx_matches) == 0:
        return pd.DataFrame()

    idx = idx_matches[0]
    # Get the position within the array (since index might not be 0-based)
    pos = list(data.index).index(idx)
    sims = pd.Series(sim_matrix[pos], index=data.index)
    sims = sims.drop(idx).sort_values(ascending=False).head(top_n)

    results = []
    for i, (row_idx, score) in enumerate(sims.items()):
        row = data.loc[row_idx]
        results.append(
            {
                "rank": i + 1,
                "name": row["name"],
                "price": row["price"],
                "rating": row.get("ratings", "N/A"),
                "ram": row.get("ram", "N/A"),
                "storage": row.get("internal_storage", "N/A"),
                "camera": row.get("primary_camera", "N/A"),
                "battery": row.get("battery_capacity", "N/A"),
                "score": score,
            }
        )
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────
df4, img_lookup = load_and_process_data()
model, model_features, metrics = train_model(df4)

# ─────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────
st.markdown(
    """
<div class="hero">
    <h1>📱 Samsung <span class="accent">SmartPick</span></h1>
    <p>AI-powered price prediction & phone recommendation engine — trained on real Samsung smartphone data</p>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# Metrics Row
# ─────────────────────────────────────────────────────────
unique_phones = df4["name"].nunique()
price_min = int(df4["price"].min())
price_max = int(df4["price"].max())

st.markdown(
    f"""
<div class="metric-row">
    <div class="metric-card">
        <p class="value">{unique_phones}</p>
        <p class="label">Phone Models</p>
    </div>
    <div class="metric-card">
        <p class="value">₹{price_min:,}</p>
        <p class="label">Starting From</p>
    </div>
    <div class="metric-card">
        <p class="value">₹{price_max:,}</p>
        <p class="label">Up To</p>
    </div>
    <div class="metric-card">
        <p class="value">{metrics["r2"]:.1%}</p>
        <p class="label">Model Accuracy (R²)</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💰  Price Predictor", "🔍  Phone Recommender"])

# ═════════════════════════════════════════════════════════
# TAB 1: PRICE PREDICTOR
# ═════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
    <div class="section-header">
        <span class="icon">⚙️</span>
        <h2>Configure Your Dream Phone</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        ram = st.select_slider("🧠 RAM (GB)", options=[2, 3, 4, 6, 8, 12, 16], value=6)
        internal_storage = st.select_slider(
            "💾 Internal Storage (GB)", options=[16, 32, 64, 128, 256, 512], value=128
        )

    with col2:
        primary_camera = st.slider(
            "📷 Primary Camera (MP)", min_value=8, max_value=200, value=50, step=2
        )
        display_size = st.slider(
            "📐 Display Size (inches)",
            min_value=4.0,
            max_value=7.5,
            value=6.5,
            step=0.1,
            format="%.1f",
        )

    with col3:
        cluster = st.selectbox(
            "🏷️ Budget Segment",
            options=["Flagship / High-End", "Mid-Range", "Budget"],
            index=1,
        )
        cluster_map = {"Flagship / High-End": 0, "Mid-Range": 1, "Budget": 2}
        cluster_val = cluster_map[cluster]

        network = st.selectbox(
            "📶 Network", options=["5G + 4G", "4G Only", "3G"], index=0
        )
        network_map = {"5G + 4G": 0, "4G Only": 1, "3G": 2}
        network_val = network_map[network]

    android = st.select_slider(
        "🤖 Android Version", options=[10, 11, 12, 13, 14, 15], value=13
    )

    st.write("")  # spacer

    if st.button("🔮  Predict Price", key="predict_btn", use_container_width=True):
        input_data = pd.DataFrame(
            {
                "internal_storage": [internal_storage],
                "ram": [ram],
                "primary_camera": [primary_camera],
                "display_size": [display_size],
                "cluster": [cluster_val],
            }
        )
        # Add optional model features
        if "Network" in model_features:
            input_data["Network"] = network_val
        if "Android 13" in model_features:
            input_data["Android 13"] = 1 if android >= 13 else 0

        # Ensure column order matches training
        input_data = input_data[model.feature_names_in_]
        predicted_price = model.predict(input_data)[0]

        st.markdown(
            f"""
        <div class="price-result">
            <p class="price-label">Estimated Market Price</p>
            <p class="price-value">₹ {predicted_price:,.0f}</p>
            <p class="price-note">Based on {metrics["n_train"]} training samples · GradientBoosting Regressor</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Show model stats
        st.markdown(
            f"""
        <div class="model-stats">
            <div class="stat-title">📊 Model Performance</div>
            <div class="stat-row">
                <span class="stat-label">R² Score</span>
                <span class="stat-value">{metrics["r2"]:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Adjusted R²</span>
                <span class="stat-value">{metrics["adj_r2"]:.4f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Mean Absolute Error</span>
                <span class="stat-value">₹ {metrics["mae"]:,.0f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Test Samples</span>
                <span class="stat-value">{metrics["n_test"]}</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════
# TAB 2: PHONE RECOMMENDER
# ═════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        """
    <div class="section-header">
        <span class="icon">🔍</span>
        <h2>Find Similar Phones</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    phone_names = sorted(df4["name"].unique().tolist())

    col_sel, col_n = st.columns([3, 1])
    with col_sel:
        selected_phone = st.selectbox(
            "Select a Samsung phone",
            phone_names,
            index=0,
            label_visibility="collapsed",
            placeholder="Search for a phone...",
        )
    with col_n:
        top_n = st.selectbox("Results", [3, 5, 8, 10], index=1)

    if st.button("🔎  Find Similar Phones", key="rec_btn", use_container_width=True):
        # Show the selected phone info
        phone_row = df4[df4["name"] == selected_phone].iloc[0]
        phone_img = img_lookup.get(selected_phone, "")

        img_tag = f'<img src="{phone_img}" alt="phone">' if phone_img else ""

        st.markdown(
            f"""
        <div class="selected-phone">
            {img_tag}
            <div class="phone-name">{selected_phone}</div>
            <div class="phone-price">₹ {phone_row["price"]:,.0f}</div>
            <div class="spec-tags" style="justify-content:center; margin-top:0.7rem;">
                <span class="spec-tag">⭐ {phone_row.get("ratings", "N/A")}</span>
                <span class="spec-tag">💾 {int(phone_row.get("internal_storage", 0))} GB</span>
                <span class="spec-tag">🧠 {int(phone_row.get("ram", 0))} GB RAM</span>
                <span class="spec-tag">📷 {int(phone_row.get("primary_camera", 0))} MP</span>
                <span class="spec-tag">🔋 {int(phone_row.get("battery_capacity", 0))} mAh</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Get recommendations
        rec_features = [
            "internal_storage",
            "ram",
            "primary_camera",
            "display_size",
            "battery_capacity",
        ]
        recs = get_recommendations(selected_phone, df4, rec_features, top_n=top_n)

        if recs.empty:
            st.warning("No recommendations found for this phone.")
        else:
            st.markdown(
                """
            <div class="section-header">
                <span class="icon">✨</span>
                <h2>Top Matches</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

            for _, rec in recs.iterrows():
                rec_img = img_lookup.get(rec["name"], "")
                img_html = (
                    f'<img class="rec-img" src="{rec_img}" alt="">' if rec_img else ""
                )
                sim_pct = rec["score"] * 100

                specs_html = ""
                if rec["ram"] != "N/A":
                    specs_html += (
                        f'<span class="spec-tag">🧠 {int(rec["ram"])} GB</span>'
                    )
                if rec["storage"] != "N/A":
                    specs_html += (
                        f'<span class="spec-tag">💾 {int(rec["storage"])} GB</span>'
                    )
                if rec["camera"] != "N/A":
                    specs_html += (
                        f'<span class="spec-tag">📷 {int(rec["camera"])} MP</span>'
                    )
                if rec["battery"] != "N/A" and rec["battery"] > 0:
                    specs_html += (
                        f'<span class="spec-tag">🔋 {int(rec["battery"])} mAh</span>'
                    )

                st.markdown(
                    f"""
                <div class="rec-card">
                    {img_html}
                    <div class="rank">{rec["rank"]}</div>
                    <div class="rec-info">
                        <div class="rec-name">{rec["name"]}</div>
                        <div class="rec-score">Similarity: {sim_pct:.1f}%</div>
                        <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{sim_pct}%"></div></div>
                        <div class="spec-tags">{specs_html}</div>
                    </div>
                    <div class="rec-price">₹ {rec["price"]:,.0f}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.25); font-size:0.8rem;'>"
    "Built with Streamlit · GradientBoosting Regressor · Cosine Similarity · Samsung Smartphone Data"
    "</p>",
    unsafe_allow_html=True,
)
