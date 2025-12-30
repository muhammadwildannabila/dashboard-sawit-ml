# =========================
# SawitScope — Ripeness Dashboard (Dark Palm: Orange • Gold • Red)
# =========================
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Plotly (visual dashboard)
import plotly.express as px
import plotly.graph_objects as go

# Optional heavy deps (UI tetap hidup walau env belum lengkap)
try:
    import joblib
except Exception:
    joblib = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import torch
    import timm
    import torchvision.transforms as T
except Exception:
    torch = None
    timm = None
    T = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="SawitScope • Ripeness", page_icon="🌴", layout="wide", initial_sidebar_state="expanded")


# =========================================================
# THEME — DARK PALM (Orange • Gold • Red) + subtle green for Mentah
# =========================================================
BG = "#050B14"
BG2 = "#07121B"
CARD = "rgba(255,255,255,0.06)"
STROKE = "rgba(255,200,80,0.16)"
TEXT = "rgba(245,248,255,0.92)"
MUTED = "rgba(245,248,255,0.68)"

ORANGE = "#FF7A1A"
GOLD = "#FFB000"
RED = "#FF3B4E"
GREEN = "#2EE59D"  # hanya untuk kelas Mentah biar “kebun” terasa, tapi aksen UI tetap orange/gold.

# Map warna kelas (konsisten di semua chart!)
CLASS_LABELS = ["Mentah", "Matang", "Busuk"]
CLASS_COLOR = {"Mentah": GREEN, "Matang": ORANGE, "Busuk": RED}

SPLIT_COLOR = {"Train": ORANGE, "Validation": "rgba(255,255,255,0.35)", "Test": GOLD}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');

* {{ font-family: 'Inter', sans-serif; }}

.stApp {{
  background:
    radial-gradient(1200px 700px at 15% 10%, rgba(255,176,0,0.22), transparent 60%),
    radial-gradient(900px 600px at 85% 20%, rgba(255,106,0,0.20), transparent 60%),
    radial-gradient(900px 700px at 60% 85%, rgba(255,200,80,0.16), transparent 65%),
    linear-gradient(180deg, #0E0B08, #070503);
  color: {TEXT};
}}

.block-container {{
  padding-top: 1.0rem;
  padding-bottom: 2.2rem;
}}

/* =====================================
   SIDEBAR — DARK PALM ORANGE / GOLD
===================================== */

[data-testid="stSidebar"] {{
  background:
    radial-gradient(
      600px 420px at 18% 8%,
      rgba(255,176,0,0.30),
      transparent 60%
    ),
    radial-gradient(
      520px 420px at 85% 88%,
      rgba(255,106,0,0.22),
      transparent 65%
    ),
    linear-gradient(
      180deg,
      #1C140A 0%,
      #120C06 60%,
      #070503 100%
    ) !important;

  border-right: 2px solid rgba(255,176,0,0.45);
  box-shadow:
    inset -10px 0 26px rgba(255,176,0,0.20),
    6px 0 28px rgba(0,0,0,0.55);
  backdrop-filter: blur(16px);
}}

/* =====================================
   SIDEBAR ITEM BASE
===================================== */

[data-testid="stSidebar"] label {{
  padding: 8px 12px;
  margin-bottom: 6px;
  border-radius: 12px;
  transition: all 0.25s ease;
}}

/* =====================================
   SIDEBAR HOVER GLOW 🌴
===================================== */

[data-testid="stSidebar"] label:hover {{
  background: linear-gradient(
    90deg,
    rgba(255,176,0,0.22),
    rgba(255,122,26,0.10)
  );
  box-shadow:
    inset 0 0 0 1px rgba(255,176,0,0.35),
    0 0 14px rgba(255,176,0,0.45);
  transform: translateX(2px);
}}

/* =====================================
   SIDEBAR ACTIVE ITEM (SELECTED)
===================================== */

[data-testid="stSidebar"] input:checked ~ div {{
  background: linear-gradient(
    90deg,
    rgba(255,176,0,0.38),
    rgba(255,122,26,0.16)
  );
  border-left: 4px solid #FFB000;
  box-shadow:
    inset 0 0 0 1px rgba(255,176,0,0.45),
    0 0 16px rgba(255,176,0,0.55);
  font-weight: 700;
}}


a {{
  color: rgba(255,176,0,0.95) !important;
  text-decoration: none;
}}

hr {{
  border: none;
  border-top: 1px solid rgba(255,200,80,0.14);
  margin: 1.2rem 0;
}}

.hero {{
  border-radius: 22px;
  padding: 18px 18px;
  border: 1px solid rgba(255,200,80,0.18);
  background:
    radial-gradient(900px 240px at 12% 0%, rgba(255,122,26,0.22), transparent 60%),
    radial-gradient(900px 260px at 85% 5%, rgba(255,176,0,0.18), transparent 60%),
    radial-gradient(800px 240px at 70% 110%, rgba(255,59,78,0.14), transparent 60%),
    linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.02));
  box-shadow: 0 12px 42px rgba(0,0,0,0.55);
}}

.h1 {{
  font-size: 34px;
  font-weight: 900;
  margin: 0;
  letter-spacing: 0.2px;
}}

.sub {{
  margin-top: 6px;
  color: {MUTED};
  font-size: 14px;
}}

.pills {{
  margin-top: 10px;
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}

.pill {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,200,80,0.22);
  background: rgba(255,176,0,0.08);
  font-size: 12px;
  color: {TEXT};
}}

.card {{
  background: linear-gradient(180deg, rgba(255,176,0,0.10), rgba(255,255,255,0.03));
  border: 1px solid rgba(255,176,0,0.35);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 34px rgba(0,0,0,0.40);
}}

.cardTitle {{
  font-weight: 800;
  font-size: 14px;
  color: {TEXT};
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}}

.small {{
  font-size: 13px;
  color: {MUTED};
}}

.kpi {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0,1fr));
  gap: 10px;
}}

.kpi .k {{
  background: rgba(255,176,0,0.10);
  border: 1px solid rgba(255,176,0,0.35);
  border-radius: 16px;
  padding: 12px 12px;
}}

.kpi .k .t {{
  color: {MUTED};
  font-size: 12px;
  letter-spacing: 0.7px;
  text-transform: uppercase;
}}

.kpi .k .v {{
  font-size: 18px;
  font-weight: 900;
  margin-top: 4px;
}}

.tag {{
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 12px;
}}

.tag-ok {{
  border-color: rgba(46,229,157,0.28);
  background: rgba(46,229,157,0.10);
}}

.tag-warn {{
  border-color: rgba(255,176,0,0.32);
  background: rgba(255,176,0,0.10);
}}

.tag-danger {{
  border-color: rgba(255,59,78,0.30);
  background: rgba(255,59,78,0.12);
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# =========================================================
# DATASET META (sesuai project kamu)
# =========================================================
DATASET = {
    "total": 3000,
    "per_class": {"Mentah": 1000, "Matang": 1000, "Busuk": 1000},
    "split": {"Train": 2100, "Validation": 450, "Test": 450},  # 70/15/15
    "desc": (
        "Dataset primer citra tandan kelapa sawit, diambil mandiri di perkebunan selama ±1 minggu. "
        "Terdiri dari 3 kelas: Mentah, Matang, Busuk (masing-masing 1000 citra)."
    )
}

# =========================================================
# MODEL RESULTS (angka dari evaluasi kamu — bukan gambar)
# =========================================================
RESULTS: Dict[str, Dict[str, Any]] = {
    "XGBoost + HSV Color": {
        "type": "xgb",
        "acc": 0.9711,
        "macro_f1": 0.9711,
        "macro_prec": 0.9712,
        "macro_rec": 0.9711,
        "train_time_min": 2.10,
        "infer_time_s_450": 0.3000,
        "cm": np.array([[147, 3, 0],
                        [  2,145, 3],
                        [  1, 4,145]]),
        "per_class": [
            ("Mentah", 0.98, 0.98, 0.98, 150),
            ("Matang", 0.95, 0.97, 0.96, 150),
            ("Busuk",  0.98, 0.97, 0.97, 150),
        ],
        "history": {
            "epoch": list(range(1, 11)),
            "train_acc": [0.900,0.940,0.955,0.965,0.972,0.978,0.982,0.986,0.988,0.990],
            "val_acc":   [0.880,0.920,0.938,0.948,0.955,0.960,0.965,0.968,0.970,0.972]
        },
        "badge": "⚡ Super cepat",
    },
    "EfficientNet-B0 + LoRA": {
        "type": "tf",
        "acc": 0.9778,
        "macro_f1": 0.9778,
        "macro_prec": 0.9778,
        "macro_rec": 0.9778,
        "train_time_min": None,
        "infer_time_s_450": None,
        "cm": np.array([[148, 2, 0],
                        [  2,146, 2],
                        [  1, 3,146]]),
        "per_class": [
            ("Mentah", 0.9801,0.9867,0.9834,150),
            ("Matang", 0.9669,0.9733,0.9701,150),
            ("Busuk",  0.9865,0.9733,0.9799,150),
        ],
        "history": {
            "epoch": list(range(1, 18)),
            "train_acc": [0.74,0.81,0.86,0.89,0.92,0.945,0.96,0.97,0.976,0.981,0.984,0.986,0.988,0.989,0.990,0.991,0.992],
            "val_acc":   [0.73,0.80,0.85,0.88,0.91,0.94,0.955,0.968,0.974,0.977,0.979,0.980,0.980,0.980,0.980,0.980,0.980],
        },
        "badge": "⚖️ Balanced",
    },
    "MaxViT-T + LoRA": {
        "type": "torch",
        "acc": 0.9867,
        "macro_f1": 0.9867,
        "macro_prec": 0.9868,
        "macro_rec": 0.9867,
        "train_time_min": None,
        "infer_time_s_450": None,
        "cm": np.array([[148, 2, 0],
                        [  1,148, 1],
                        [  0, 2,148]]),
        "per_class": [
            ("Mentah", 0.9933,0.9867,0.9900,150),
            ("Matang", 0.9737,0.9867,0.9801,150),
            ("Busuk",  0.9933,0.9867,0.9900,150),
        ],
        "history": {
            "epoch": list(range(1, 20)),
            "train_acc": [0.75,0.82,0.87,0.90,0.93,0.95,0.962,0.973,0.980,0.985,0.987,0.989,0.990,0.991,0.992,0.9925,0.993,0.9935,0.994],
            "val_acc":   [0.74,0.81,0.86,0.89,0.92,0.945,0.962,0.972,0.978,0.983,0.986,0.9865,0.987,0.987,0.987,0.987,0.987,0.987,0.987],
        },
        "badge": "🏆 Best accuracy",
    },
}

MODEL_ORDER = ["MaxViT-T + LoRA", "EfficientNet-B0 + LoRA", "XGBoost + HSV Color"]


# =========================================================
# FILES (model artifacts untuk inferensi)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent   # .../src
MODELS_DIR = BASE_DIR.parent / "sawit_models"

FILES = {
    "class_names": MODELS_DIR / "class_names.json",
    "xgb_model": MODELS_DIR / "xgb_hsv.joblib",
    "xgb_meta": MODELS_DIR / "xgb_meta.joblib",
    "effnet": MODELS_DIR / "model_effnetb0_lora_merged.keras",
    "maxvit": MODELS_DIR / "maxvit_merged.pt",
}

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png"}


# =========================================================
# Small helpers (biar Streamlit future-proof dari warning use_container_width)
# =========================================================
def _img_show(img, caption=None):
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)

def _df_show(df: pd.DataFrame):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


# =========================================================
# UTIL
# =========================================================
def safe_label(lbl: str) -> str:
    mapping = {"unripe": "Mentah", "ripe": "Matang", "rotten": "Busuk"}
    return mapping.get(lbl, lbl)

def extract_images_from_zip(zip_bytes: bytes) -> List[Tuple[str, Image.Image]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            if ext not in ALLOWED_IMG_EXT:
                continue
            with z.open(info) as f:
                try:
                    img = Image.open(f).convert("RGB")
                    out.append((Path(info.filename).name, img))
                except Exception:
                    pass
    return out

def pred_badge(label: str) -> str:
    cls = {
        "Mentah": "tag-ok",
        "Matang": "tag-warn",
        "Busuk": "tag-danger"
    }.get(label, "")
    return f'<span class="tag {cls}">🧠 {label}</span>'

@st.cache_data
def load_class_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        keys = list(data.keys())
        if all(str(k).isdigit() for k in keys):
            return [data[str(i)] for i in range(len(keys))]
        return list(data.values())
    return data

def topk(prob: np.ndarray, class_names: List[str], k: int = 3) -> List[Tuple[str, float]]:
    idx = np.argsort(prob)[::-1][:k]
    return [(class_names[i], float(prob[i])) for i in idx]

def margin_top1_top2(prob: np.ndarray) -> float:
    s = np.sort(prob)[::-1]
    return float(s[0] - s[1]) if len(s) > 1 else 0.0

def insight_tips(top1: str, top2: str) -> List[str]:
    tips = [
        "Foto ulang lebih dekat (objek memenuhi frame).",
        "Gunakan cahaya merata (hindari bayangan tajam/backlight).",
        "Pastikan fokus tajam (tidak blur) dan tandan terlihat jelas.",
        "Kurangi background ramai; pakai latar bersih/kontras."
    ]
    pair = {top1, top2}
    if {"ripe", "unripe"} <= pair:
        tips.insert(0, "Kelas transisi **Mentah ↔ Matang** sering mirip. Ambil foto di cahaya natural & jarak lebih dekat.")
    if "rotten" in pair:
        tips.insert(0, "Jika area busuk kecil/gelap kurang terlihat, model bisa ragu. Pastikan area tersebut tertangkap jelas.")
    return tips


# =========================================================
# ASSET VALIDATION (PER MODEL)
# =========================================================
def require_files(paths: List[Path]) -> List[str]:
    missing = []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
    return missing


# =========================================================
# MODEL LOADERS (inferensi)
# =========================================================
@st.cache_resource
def load_xgb():
    if joblib is None:
        raise RuntimeError("Dependency missing: joblib. Install joblib + xgboost + scikit-learn.")
    xgb = joblib.load(str(FILES["xgb_model"]))
    meta = joblib.load(str(FILES["xgb_meta"]))
    return xgb, meta

def color_features_hsv(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-6)
    s_hist = s_hist / (s_hist.sum() + 1e-6)
    v_hist = v_hist / (v_hist.sum() + 1e-6)
    mean = hsv.mean(axis=(0, 1))
    std = hsv.std(axis=(0, 1))
    return np.concatenate([h_hist, s_hist, v_hist, mean, std]).astype(np.float32)

def xgb_predict(pil_img: Image.Image, xgb, meta) -> Tuple[str, float, np.ndarray, List[str]]:
    if cv2 is None:
        raise RuntimeError("Dependency missing: opencv-python(-headless).")
    classes = meta.get("classes", [])
    img_size = tuple(meta.get("img_size", [160, 160]))

    rgb = np.array(pil_img.convert("RGB"))
    rgb = cv2.resize(rgb, img_size)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    feat = color_features_hsv(bgr).reshape(1, -1)
    prob = xgb.predict_proba(feat)[0]
    idx = int(np.argmax(prob))
    return classes[idx], float(prob[idx]), prob, classes

@st.cache_resource
def load_effnet():
    if tf is None:
        raise RuntimeError("Dependency missing: tensorflow.")
    return tf.keras.models.load_model(str(FILES["effnet"]))

def effnet_predict(pil_img: Image.Image, model, class_names: List[str]) -> Tuple[str, float, np.ndarray]:
    img = pil_img.convert("RGB").resize((160, 160))
    x = (np.array(img).astype("float32") / 255.0)[None, ...]
    prob = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(prob))
    return class_names[idx], float(prob[idx]), prob

@st.cache_resource
def load_maxvit():
    if torch is None or timm is None or T is None:
        raise RuntimeError("Dependency missing: torch + timm + torchvision.")
    ckpt = torch.load(str(FILES["maxvit"]), map_location="cpu")
    arch = ckpt["arch"]
    classes = ckpt["classes"]
    img_size = int(ckpt.get("img_size", 224))

    model = timm.create_model(arch, pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, classes, img_size

def maxvit_predict(pil_img: Image.Image, model, class_names: List[str], img_size: int) -> Tuple[str, float, np.ndarray]:
    device = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    model = model.to(device)
    tfm = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    x = tfm(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(prob))
    return class_names[idx], float(prob[idx]), prob


# =========================================================
# Plotly figure builders (Sawit palette)
# =========================================================
def fig_donut_class():
    df = pd.DataFrame({"Kelas": list(DATASET["per_class"].keys()), "Jumlah": list(DATASET["per_class"].values())})
    fig = px.pie(df, names="Kelas", values="Jumlah", hole=0.62,
                 color="Kelas", color_discrete_map=CLASS_COLOR)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(
        height=340, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        showlegend=False
    )
    return fig

def fig_split_bar():
    df = pd.DataFrame({"Split": list(DATASET["split"].keys()), "Jumlah": list(DATASET["split"].values())})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Split"], y=df["Jumlah"],
        marker=dict(color=[SPLIT_COLOR[s] for s in df["Split"]], line=dict(color="rgba(255,255,255,0.18)", width=1)),
        text=df["Jumlah"], textposition="outside",
        hovertemplate="<b>%{x}</b><br>Jumlah: %{y}<extra></extra>"
    ))
    fig.update_layout(
        height=320, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="Jumlah Citra"),
    )
    return fig

def fig_model_perf():
    df = pd.DataFrame([
        {"Model": k, "Accuracy": v["acc"]*100, "Macro-F1": v["macro_f1"]*100, "Macro-Recall": v["macro_rec"]*100}
        for k, v in RESULTS.items()
    ])
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values("Model")

    fig = go.Figure()
    for metric, color in [("Accuracy", GOLD), ("Macro-Recall", ORANGE), ("Macro-F1", RED)]:
        fig.add_trace(go.Bar(
            name=metric,
            x=df["Model"],
            y=df[metric],
            marker=dict(color=color),
            text=[f"{x:.2f}%" for x in df[metric]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>"+metric+": %{y:.2f}%<extra></extra>"
        ))
    fig.update_layout(
        barmode="group",
        height=340, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="Skor (%)", range=[90, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def fig_train_curve(model_name: str):
    h = RESULTS[model_name]["history"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["train_acc"], mode="lines+markers", name="Train",
                             line=dict(color=GOLD, width=3)))
    fig.add_trace(go.Scatter(x=h["epoch"], y=h["val_acc"], mode="lines+markers", name="Validation",
                             line=dict(color=ORANGE, width=3)))
    fig.update_layout(
        height=340, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(showgrid=False, title="Epoch/Iterasi"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", title="Accuracy", range=[0.70, 1.00]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def fig_cm(model_name: str):
    cm = RESULTS[model_name]["cm"]
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=CLASS_LABELS,
        y=CLASS_LABELS,
        colorscale="YlOrRd",
        showscale=True,
        colorbar=dict(title="Jumlah"),
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16, "color": "#9e9696"},
        hovertemplate="Actual: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(
        height=340, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        xaxis=dict(title="Prediksi"),
        yaxis=dict(title="Label Asli"),
    )
    return fig


# =========================================================
# HERO
# =========================================================
st.markdown(
    f"""
    <div class="hero">
      <div class="h1">🌴 SawitScope • Ripeness Classifier & Analytics Dashboard</div>
      <div class="sub">
        Dashboard analitik dan prediksi kematangan tandan kelapa sawit 
        (Mentah • Matang • Busuk) menggunakan tiga pendekatan utama:
        <b>Klasik</b>, <b>Transfer Learning</b>, dan <b>Transformer</b>.
      </div>

      <div class="pills">
        <span class="pill">🧪 Dataset: <b>{DATASET["total"]:,}</b> citra</span>
        <span class="pill">🧩 Split: <b>70 / 15 / 15</b></span>
        <span class="pill">🏆 Best Model: <b>MaxViT-T + LoRA</b> 
          ({RESULTS["MaxViT-T + LoRA"]["acc"]*100:.2f}%)
        </span>
        <span class="pill">⚡ Fast Model: <b>XGBoost + Color</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Sidebar — Navigasi + Kontrol
# =========================================================
if "scan_counter" not in st.session_state:
    st.session_state["scan_counter"] = 0

with st.sidebar:
    # -----------------------------------------------------
    # Brand / Header
    # -----------------------------------------------------
    st.markdown("# 🌴 SawitScope")
    st.caption("By: Muhammad Wildan Nabila")
    st.markdown("---")

    # -----------------------------------------------------
    # Navigasi Utama
    # -----------------------------------------------------
    st.markdown("### 🧭 Navigasi")
    nav = st.radio(
        label="Menu",
        options=[
            "🏠 Overview",
            "🧭 Panduan Pengguna",
            "🔮 Prediksi Tunggal",
            "📦 Prediksi Massal",
            "⚔️ Perbandingan Model",
            "📊 Insight & Analisis",
            "ℹ️ Informasi Proyek"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # Model Selection
    # -----------------------------------------------------
    st.subheader("🤖 Model Aktif")
    model_choice = st.selectbox(
        "Pilih model untuk inferensi",
        MODEL_ORDER,
        index=1,
        help="Model ini akan digunakan untuk prediksi gambar"
    )

    st.markdown("---")

    # -----------------------------------------------------
    # Prediction Controls
    # -----------------------------------------------------
    st.subheader("🎛️ Kontrol Prediksi")

    conf_th = st.slider(
        "Confidence minimum",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
        help="Prediksi di bawah threshold akan ditandai sebagai berisiko"
    )

    margin_th = st.slider(
        "Ambiguity margin (Top1–Top2)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="Margin kecil menandakan kelas saling berdekatan"
    )

    show_top3 = st.checkbox(
        "Tampilkan Top-3 prediksi",
        value=True
    )

    show_insight = st.checkbox(
        "Tampilkan Insight otomatis",
        value=True
    )

    preview_limit = st.slider(
        "Jumlah preview gambar",
        min_value=3,
        max_value=18,
        value=9,
        step=1
    )

    st.markdown("---")

    # -----------------------------------------------------
    # Quick Stats
    # -----------------------------------------------------
    st.subheader("📈 Quick Stats")

    st.metric(
        label="Total Dataset",
        value=f"{DATASET['total']:,}",
        delta="images"
    )

    st.metric(
        label="Session Predictions",
        value=f"{st.session_state['scan_counter']:,}",
        delta="+running"
    )

# =========================================================
# Load class names for TF display (wajib untuk effnet)
# =========================================================
missing_base = require_files([FILES["class_names"]])
if missing_base:
    st.warning("File `class_names.json` belum ada. Dashboard tetap jalan, tapi inferensi EffNet butuh file ini.")
    class_names_global = ["unripe", "ripe", "rotten"]
else:
    class_names_global = load_class_names(FILES["class_names"])


# =========================================================
# Load model sesuai pilihan (lazy)
# =========================================================
err_model = None
xgb_obj = meta_obj = None
eff_obj = None
maxvit_obj = maxvit_classes = None
maxvit_img = 224

need = [FILES["class_names"]]
if model_choice.startswith("XGBoost"):
    need += [FILES["xgb_model"], FILES["xgb_meta"]]
elif model_choice.startswith("EfficientNet"):
    need += [FILES["effnet"]]
else:
    need += [FILES["maxvit"]]

missing = require_files([p for p in need if p is not None])
# kalau model file belum ada, jangan stop total—stop hanya di halaman predict/batch/arena
MODEL_READY = (len(missing) == 0)


def predict_one(pil_img: Image.Image) -> Dict[str, Any]:
    if not MODEL_READY:
        raise RuntimeError("Model belum lengkap. Lengkapi file model di folder sawit_models/.")

    if model_choice.startswith("XGBoost"):
        if xgb_obj is None:
            raise RuntimeError("XGB not loaded")
        label, conf, prob, cn = xgb_predict(pil_img, xgb_obj, meta_obj)
        class_names = cn
    elif model_choice.startswith("EfficientNet"):
        if eff_obj is None:
            raise RuntimeError("EffNet not loaded")
        label, conf, prob = effnet_predict(pil_img, eff_obj, class_names_global)
        class_names = class_names_global
    else:
        if maxvit_obj is None:
            raise RuntimeError("MaxViT not loaded")
        label, conf, prob = maxvit_predict(pil_img, maxvit_obj, maxvit_classes, maxvit_img)
        class_names = maxvit_classes

    prob = np.array(prob, dtype=float)
    idx_sorted = np.argsort(prob)[::-1]
    top1 = (class_names[int(idx_sorted[0])], float(prob[int(idx_sorted[0])]))
    top2 = (class_names[int(idx_sorted[1])], float(prob[int(idx_sorted[1])])) if len(prob) > 1 else ("-", 0.0)
    m = float(top1[1] - top2[1])

    low_conf = float(conf) < conf_th
    ambiguous = m < margin_th

    return {
        "pred_label": safe_label(label),
        "confidence": float(conf),
        "margin": m,
        "top3": [(safe_label(lbl), p) for (lbl, p) in topk(prob, class_names, k=3)],
        "low_conf": low_conf,
        "ambiguous": ambiguous,
    }


# =========================================================
# COMMON: GET INPUT ITEMS
# =========================================================
def get_items(mode_choice: str) -> List[Tuple[str, Image.Image]]:
    items: List[Tuple[str, Image.Image]] = []
    if mode_choice == "Single / Multi Image":
        files = st.file_uploader("Upload gambar (bisa lebih dari 1)", type=["jpg","jpeg","png"], accept_multiple_files=True)
        if files:
            for f in files:
                try:
                    items.append((f.name, Image.open(f).convert("RGB")))
                except Exception:
                    pass
    else:
        zf = st.file_uploader("Upload ZIP berisi gambar", type=["zip"])
        if zf is not None:
            items = extract_images_from_zip(zf.read())
    return items


# =========================================================
# Load model only when needed (Predict/Batch/Arena)
# =========================================================
def ensure_model_loaded():
    global xgb_obj, meta_obj, eff_obj, maxvit_obj, maxvit_classes, maxvit_img, err_model, MODEL_READY

    if not MODEL_READY:
        return

    try:
        if model_choice.startswith("XGBoost") and xgb_obj is None:
            xgb_obj, meta_obj = load_xgb()
        elif model_choice.startswith("EfficientNet") and eff_obj is None:
            eff_obj = load_effnet()
        elif model_choice.startswith("MaxViT") and maxvit_obj is None:
            maxvit_obj, maxvit_classes, maxvit_img = load_maxvit()
    except Exception as e:
        err_model = str(e)
        MODEL_READY = False


# =========================================================
# PAGE: DASHBOARD
# =========================================================
if nav == "🏠 Overview":
    st.markdown('<div class="card"><div class="cardTitle">📌 Ringkasan Project</div><div class="small">'
                'Tampilan awal dashboard dirancang agar pengguna langsung memahami gambaran sistem, meliputi dataset, split, performa model, dan evaluasi utama.</div></div>',
                unsafe_allow_html=True)
    st.write("")

    best_name = max(RESULTS.keys(), key=lambda k: RESULTS[k]["acc"])
    best = RESULTS[best_name]

    # KPI row
    st.markdown(
        f"""
<div class="kpi">
  <div class="k"><div class="t">Dataset Size</div><div class="v">{DATASET["total"]:,}</div></div>
  <div class="k"><div class="t">Classes</div><div class="v">3 Kelas (Distribusi Merata)</div></div>
  <div class="k"><div class="t">Split</div><div class="v">70/15/15</div></div>
  <div class="k"><div class="t">Best Model</div><div class="v">{best_name}</div></div>
</div>
""",
        unsafe_allow_html=True
    )
    st.write("")

    # Row 1
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<div class="card"><div class="cardTitle">🧃 Distribusi Kelas (Balanced)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_donut_class(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card"><div class="cardTitle">🏁 Performa Model (Test Set)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_model_perf(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    c3, c4 = st.columns([1, 1], gap="large")
    with c3:
        st.markdown('<div class="card"><div class="cardTitle">🧩 Split Dataset (70/15/15)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_split_bar(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="card"><div class="cardTitle">🔥 Confusion Matrix — Best Model</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_cm(best_name), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Status line
    st.markdown('<div class="card"><div class="cardTitle">🧠 Highlight</div>', unsafe_allow_html=True)
    st.write(f"- **Dataset**: {DATASET['total']} citra (Mentah/Matang/Busuk merata).")
    st.write(f"- **Best accuracy**: **{best_name}** = **{best['acc']*100:.2f}%** (macro-F1: {best['macro_f1']*100:.2f}%).")
    st.write("- Untuk deployment CPU umum: **EfficientNet-B0 + LoRA** biasanya paling *balanced*.")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "🧭 Panduan Pengguna":
    st.markdown("## 📘 Tata Cara Menggunakan Dashboard SawitScope")
    st.caption("Panduan interaktif langkah demi langkah")

    st.markdown("---")

    # =========================
    # INIT STATE
    # =========================
    TOTAL_STEPS = 6

    if "guide_step" not in st.session_state:
        st.session_state.guide_step = 1

    # =========================
    # PROGRESS BAR
    # =========================
    progress_value = st.session_state.guide_step / TOTAL_STEPS
    st.progress(progress_value)

    st.markdown(
        f"**Step {st.session_state.guide_step} dari {TOTAL_STEPS}**"
    )

    st.markdown("---")

    # =========================
    # STEP CONTENT
    # =========================
    step = st.session_state.guide_step

    if step == 1:
        st.markdown("### 📊 Step 1 — Kenali Dashboard")
        st.info("""
        - Ringkasan dataset sawit
        - Distribusi kelas: **Mentah, Matang, Busuk**
        - Statistik awal & performa model
        """)

    elif step == 2:
        st.markdown("### 🤖 Step 2 — Pilih Model")
        st.info("""
        - Gunakan **sidebar**
        - Pilih model:
          - XGBoost + Warna
          - EfficientNet-B0 + LoRA
          - MaxViT-T + LoRA
        """)

    elif step == 3:
        st.markdown("### 🔮 Step 3 — Prediksi Gambar")
        st.info("""
        1. Masuk menu **Predict**
        2. Upload 1 gambar sawit
        3. Lihat hasil prediksi & confidence
        """)

    elif step == 4:
        st.markdown("### 📦 Step 4 — Batch Prediction")
        st.info("""
        - Upload banyak gambar
        - Cocok untuk evaluasi dataset
        """)

    elif step == 5:
        st.markdown("### 📈 Step 5 — Analisis & Insight")
        st.info("""
        - Bandingkan performa model
        - Confusion Matrix & Classification Report
        """)

    elif step == 6:
        st.markdown("### ✅ Step 6 — Selesai")
        st.success("""
        🎉 Dashboard siap digunakan!

        👉 Disarankan mulai dari menu **Predict**
        """)

    st.markdown("---")

    # =========================
    # NAVIGATION BUTTONS
    # =========================
    col_prev, col_space, col_next = st.columns([2, 6, 2])

    with col_prev:
        if st.session_state.guide_step > 1:
            if st.button("⬅️ Back", use_container_width=True):
                st.session_state.guide_step -= 1
                st.rerun()

    with col_next:
        if st.session_state.guide_step < TOTAL_STEPS:
            if st.button("➡️ Next", use_container_width=True):
                st.session_state.guide_step += 1
                st.rerun()


# =========================================================
# PAGE: PREDICT
# =========================================================
elif nav == "🔮 Prediksi Tunggal":
    st.markdown('<div class="card"><div class="cardTitle">🔮 Prediksi Kematangan</div>'
                '<div class="small">Upload gambar → lihat label, confidence, margin Top1–Top2, dan Top-3.</div></div>',
                unsafe_allow_html=True)
    st.write("")

    if not MODEL_READY:
        st.error("Model/dependency belum siap untuk inferensi.")
        if missing:
            st.write("Missing files:")
            for m in missing:
                st.write(f"- {m}")
        st.stop()

    ensure_model_loaded()
    if err_model:
        st.error("Gagal load model/dependency.")
        st.code(err_model)
        st.stop()

    mode = st.radio("Mode input", ["Single / Multi Image", "ZIP Batch"], horizontal=True, index=0)

    left, right = st.columns([1.05, 1.0], gap="large")
    with left:
        st.markdown('<div class="card"><div class="cardTitle">📥 Input</div>', unsafe_allow_html=True)
        items = get_items(mode)
        if not items:
            st.info("Upload gambar/ZIP untuk memulai.")
        else:
            st.markdown(f"**Preview (maks {preview_limit} gambar)**")
            cols = st.columns(3)
            for i, (name, img) in enumerate(items[:preview_limit]):
                with cols[i % 3]:
                    _img_show(img, caption=name)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="cardTitle">🧠 Output</div>', unsafe_allow_html=True)
        items = items if "items" in locals() else []
        if not items:
            st.warning("Belum ada input.")
        else:
            st.session_state["scan_counter"] += len(items)

            rows = []
            risky = []
            for name, img in items:
                out = predict_one(img)
                rows.append({
                    "filename": name,
                    "pred": out["pred_label"],
                    "confidence": round(out["confidence"], 4),
                    "margin": round(out["margin"], 4),
                    "low_conf": bool(out["low_conf"]),
                    "ambiguous": bool(out["ambiguous"]),
                    "top3": ", ".join([f"{k}:{p:.3f}" for k, p in out["top3"]]) if show_top3 else ""
                })
                if out["low_conf"] or out["ambiguous"]:
                    risky.append((name, img, out))

            df = pd.DataFrame(rows).sort_values(["confidence", "margin"], ascending=False)
            low_cnt = int(df["low_conf"].sum())
            amb_cnt = int(df["ambiguous"].sum())

            st.markdown(
                f"""
<div class="kpi">
  <div class="k"><div class="t">Model</div><div class="v">{model_choice}</div></div>
  <div class="k"><div class="t">Total</div><div class="v">{len(df)}</div></div>
  <div class="k"><div class="t">Low confidence</div><div class="v">{low_cnt}</div></div>
  <div class="k"><div class="t">Ambiguous</div><div class="v">{amb_cnt}</div></div>
</div>
""",
                unsafe_allow_html=True
            )
            st.write("")
            _df_show(df)

            # Distribusi prediksi
            vc = df["pred"].value_counts().reindex(CLASS_LABELS, fill_value=0).reset_index()
            vc.columns = ["Kelas", "Jumlah"]
            fig = px.bar(vc, x="Kelas", y="Jumlah", color="Kelas",
                         color_discrete_map=CLASS_COLOR)
            fig.update_layout(
                height=300, margin=dict(l=10,r=10,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", data=csv, file_name="prediksi_sawit_scope.csv", mime="text/csv", use_container_width=True)

            if show_insight:
                st.markdown("---")
                st.markdown("#### 🔍 Insight Risiko (tanpa ground-truth)")
                if not risky:
                    st.markdown('<span class="tag tag-ok">✅ Prediksi stabil (confidence & margin aman)</span>', unsafe_allow_html=True)
                else:
                    risky.sort(key=lambda t: (t[2]["margin"], t[2]["confidence"]))
                    name0, img0, out0 = risky[0]
                    st.markdown('<span class="tag tag-danger">⚠️ Ada prediksi berisiko</span>', unsafe_allow_html=True)
                    _img_show(img0, caption=f"Contoh paling ragu: {name0}")
                    
                    st.markdown(
                        pred_badge(out0["pred_label"]),
                        unsafe_allow_html=True
                    )
                    
                    st.write(
                        f"**Confidence:** {out0['confidence']:.3f} | "
                        f"**Margin (Top1–Top2):** {out0['margin']:.3f}"
                    )          
                    st.write("**Saran cepat:**")
                    for tip in insight_tips(out0["top3"][0][0], out0["top3"][1][0] if len(out0["top3"]) > 1 else out0["top3"][0][0]):
                        st.write(f"- {tip}")

        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE: BATCH
# =========================================================
elif nav == "📦 Prediksi Massal":
    st.markdown('<div class="card"><div class="cardTitle">📦 Batch Inspector</div>'
                '<div class="small">Upload banyak gambar/ZIP → rekap hasil → download CSV.</div></div>',
                unsafe_allow_html=True)
    st.write("")

    if not MODEL_READY:
        st.error("Model/dependency belum siap untuk inferensi batch.")
        if missing:
            for m in missing:
                st.write(f"- {m}")
        st.stop()

    ensure_model_loaded()
    if err_model:
        st.error("Gagal load model/dependency.")
        st.code(err_model)
        st.stop()

    batch_mode = st.radio("Pilih mode batch", ["Multi Image", "ZIP Batch"], index=1, horizontal=True)
    items: List[Tuple[str, Image.Image]] = []

    if batch_mode == "Multi Image":
        files = st.file_uploader("Upload banyak gambar", type=["jpg","jpeg","png"], accept_multiple_files=True)
        if files:
            for f in files:
                try:
                    items.append((f.name, Image.open(f).convert("RGB")))
                except Exception:
                    pass
    else:
        zf = st.file_uploader("Upload ZIP berisi gambar", type=["zip"])
        if zf is not None:
            items = extract_images_from_zip(zf.read())

    if not items:
        st.info("Upload dulu agar muncul hasil batch.")
        st.stop()

    st.session_state["scan_counter"] += len(items)

    st.markdown('<div class="card"><div class="cardTitle">🖼️ Preview</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (name, img) in enumerate(items[:preview_limit]):
        with cols[i % 3]:
            _img_show(img, caption=name)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")

    rows = []
    for name, img in items:
        out = predict_one(img)
        rows.append({
            "filename": name,
            "pred": out["pred_label"],
            "confidence": round(out["confidence"], 4),
            "margin": round(out["margin"], 4),
            "low_conf": bool(out["low_conf"]),
            "ambiguous": bool(out["ambiguous"]),
            "top3": ", ".join([f"{k}:{p:.3f}" for k, p in out["top3"]]) if show_top3 else ""
        })

    df = pd.DataFrame(rows).sort_values(["confidence","margin"], ascending=False)
    low_cnt = int(df["low_conf"].sum())
    amb_cnt = int(df["ambiguous"].sum())

    st.markdown('<div class="card"><div class="cardTitle">📋 Ringkasan</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi">
  <div class="k"><div class="t">Total</div><div class="v">{len(df)}</div></div>
  <div class="k"><div class="t">Low confidence</div><div class="v">{low_cnt}</div></div>
  <div class="k"><div class="t">Ambiguous</div><div class="v">{amb_cnt}</div></div>
  <div class="k"><div class="t">Model</div><div class="v">{model_choice}</div></div>
</div>
""",
        unsafe_allow_html=True
    )
    st.write("")
    _df_show(df)

    vc = df["pred"].value_counts().reindex(CLASS_LABELS, fill_value=0).reset_index()
    vc.columns = ["Kelas", "Jumlah"]
    fig = px.bar(vc, x="Kelas", y="Jumlah", color="Kelas", color_discrete_map=CLASS_COLOR)
    fig.update_layout(
        height=280, margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT), showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    )
    st.plotly_chart(fig, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV Batch", data=csv, file_name="prediksi_batch_sawit.csv", mime="text/csv", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE: MODEL ARENA (compare 3 models on same image)
# =========================================================
elif nav == "⚔️ Perbandingan Model":
    st.markdown('<div class="card"><div class="cardTitle">⚔️ Model Arena</div>'
                '<div class="small">Tes 1 gambar → lihat prediksi 3 model sekaligus (XGB vs EffNet vs MaxViT).</div></div>',
                unsafe_allow_html=True)
    st.write("")

    up = st.file_uploader("Upload 1 gambar untuk arena", type=["jpg","jpeg","png"])
    if not up:
        st.info("Upload dulu untuk memulai arena.")
        st.stop()

    img = Image.open(up).convert("RGB")
    _img_show(img, caption="Arena Image")
    st.write("")

    # try load all models that exist
    arena_models = [
        ("XGBoost + HSV Color", "xgb"),
        ("EfficientNet-B0 + LoRA", "tf"),
        ("MaxViT-T + LoRA", "torch"),
    ]

    if st.button("⚔️ START ARENA", use_container_width=True):
        results_arena = []

        # XGB
        try:
            if require_files([FILES["xgb_model"], FILES["xgb_meta"]]) == []:
                xgb, meta = load_xgb()
                label, conf, prob, cn = xgb_predict(img, xgb, meta)
                results_arena.append(("XGBoost + HSV Color", safe_label(label), float(conf), float(margin_top1_top2(prob))))
        except Exception as e:
            results_arena.append(("XGBoost + HSV Color", "ERROR", 0.0, 0.0))

        # EffNet
        try:
            if require_files([FILES["effnet"], FILES["class_names"]]) == []:
                eff = load_effnet()
                label, conf, prob = effnet_predict(img, eff, class_names_global)
                results_arena.append(("EfficientNet-B0 + LoRA", safe_label(label), float(conf), float(margin_top1_top2(prob))))
        except Exception as e:
            results_arena.append(("EfficientNet-B0 + LoRA", "ERROR", 0.0, 0.0))

        # MaxViT
        try:
            if require_files([FILES["maxvit"]]) == []:
                mv, mv_classes, mv_img = load_maxvit()
                label, conf, prob = maxvit_predict(img, mv, mv_classes, mv_img)
                results_arena.append(("MaxViT-T + LoRA", safe_label(label), float(conf), float(margin_top1_top2(prob))))
        except Exception as e:
            results_arena.append(("MaxViT-T + LoRA", "ERROR", 0.0, 0.0))

        st.session_state["scan_counter"] += 3

        df = pd.DataFrame(results_arena, columns=["Model", "Prediksi", "Confidence", "Margin"])
        st.markdown('<div class="card"><div class="cardTitle">🥊 Hasil Arena</div>', unsafe_allow_html=True)
        _df_show(df)
        st.markdown('</div>', unsafe_allow_html=True)

        # chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["Model"],
            y=df["Confidence"],
            marker=dict(color=[GOLD, ORANGE, RED]),
            text=[f"{v:.3f}" for v in df["Confidence"]],
            textposition="outside"
        ))
        fig.update_layout(
            height=320, margin=dict(l=10,r=10,t=10,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            yaxis=dict(range=[0,1], showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: ANALYTICS (no images; all generated)
# =========================================================
elif nav == "📊 Insight & Analisis":
    st.markdown('<div class="card"><div class="cardTitle">📊 Analytics</div>'
                '<div class="small">Seluruh visualisasi analitik disusun berdasarkan data numerik dan array evaluasi (tanpa menggunakan gambar hasil prediksi langsung). CM, per-class metrics, dan train curve.</div></div>',
                unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3 = st.tabs(["🔥 Confusion Matrix", "📈 Train Curves", "🎯 Per-Class Metrics"])

    with tab1:
        pick = st.selectbox("Pilih model (CM)", MODEL_ORDER, index=0, key="cm_pick")
        st.markdown('<div class="card"><div class="cardTitle">🔥 Confusion Matrix</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_cm(pick), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        pick = st.selectbox("Pilih model (Curve)", MODEL_ORDER, index=0, key="curve_pick")
        st.markdown('<div class="card"><div class="cardTitle">📈 Train vs Validation Accuracy</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_train_curve(pick), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        pick = st.selectbox("Pilih model (Per-class)", MODEL_ORDER, index=0, key="pc_pick")
        rows = []
        for (kls, p, r, f1, sup) in RESULTS[pick]["per_class"]:
            rows.append({"Kelas": kls, "Precision": p, "Recall": r, "F1": f1, "Support": sup})
        df = pd.DataFrame(rows)

        st.markdown('<div class="card"><div class="cardTitle">🎯 Per-Class Report (Test)</div>', unsafe_allow_html=True)
        _df_show(df)

        # radar-like polar chart
        fig = go.Figure()
        for metric, col in [("Precision", GOLD), ("Recall", ORANGE), ("F1", RED)]:
            fig.add_trace(go.Scatterpolar(
                r=df[metric].tolist(),
                theta=df["Kelas"].tolist(),
                fill="toself",
                name=metric,
                line=dict(color=col)
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0.90, 1.0])),
            showlegend=True,
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            margin=dict(l=10,r=10,t=10,b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE: ABOUT (sesuai teks kamu)
# =========================================================
elif nav == "ℹ️ Informasi Proyek":
    st.markdown('<div class="card"><div class="cardTitle">ℹ️ Deskripsi Dataset & EDA</div></div>', unsafe_allow_html=True)
    st.write("")

    st.markdown('<div class="card"><div class="cardTitle">1) Original Data</div>', unsafe_allow_html=True)
    st.write(
        "Dataset yang digunakan berupa **citra buah kelapa sawit** untuk klasifikasi tingkat kematangan. "
        "Data merupakan **data primer**, dikumpulkan secara mandiri di perkebunan sawit selama ±1 minggu. "
        f"Total dataset **{DATASET['total']}** citra dengan **3 kelas**: "
        "**Mentah**, **Matang**, dan **Busuk**, masing-masing sebanyak **1000** citra."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><div class="cardTitle">2) EDA • Distribusi Jumlah per Kelas</div>', unsafe_allow_html=True)
    st.write(
        "EDA menampilkan distribusi jumlah citra pada setiap kelas menggunakan visualisasi statistik. "
        "Hasil analisis menunjukkan bahwa dataset bersifat **merata dan seimbang** di antara ketiga kelas."
    )
    st.plotly_chart(fig_donut_class(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><div class="cardTitle">3) Split Data</div>', unsafe_allow_html=True)
    st.write(
        "Dataset dibagi menjadi **Train/Validation/Test = 70%/15%/15%** "
        f"(Train={DATASET['split']['Train']}, Validation={DATASET['split']['Validation']}, Test={DATASET['split']['Test']})."
    )
    st.plotly_chart(fig_split_bar(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><div class="cardTitle">4) Catatan Implementasi</div>', unsafe_allow_html=True)
    st.write("- Warna kelas konsisten: **Mentah=Hijau**, **Matang=Oranye**, **Busuk=Merah**.")
    st.write("- Semua chart dibuat dari angka/array → **tanpa gambar result**.")
    st.markdown('</div>', unsafe_allow_html=True)


st.caption("SawitScope • By: Muhammad Wildan Nabila • Streamlit + Plotly")
