"""
RapidRelief AI — Aplicación web de clasificación textil
Streamlit app para demostración y validación del modelo en entorno de desarrollo.
"""

import os
import time
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
import plotly.graph_objects as go

# ─── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="RapidRelief AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "RapidRelief AI\n"
            "Clasificación automatizada de donaciones textiles para respuesta humanitaria.\n\n"
            "Equipo: Fernando Acuña · Pamela Ruíz · Dijo Lozada\n"
            "Instructor: Dr. José Ambrosio Bastián · USB 2026"
        )
    },
)

# ─── Constantes ───────────────────────────────────────────────────────────────
CLASES = [
    ("Vestido",     "dress",      "01"),
    ("Sombrero",    "hat",        "02"),
    ("Manga larga", "longsleeve", "03"),
    ("Abrigo",      "outwear",    "04"),
    ("Pantalón",    "pants",      "05"),
    ("Camisa",      "shirt",      "06"),
    ("Zapatos",     "shoes",      "07"),
    ("Shorts",      "shorts",     "08"),
    ("Falda",       "skirt",      "09"),
    ("Camiseta",    "t-shirt",    "10"),
]

IMAGE_SIZE = (224, 224)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model/clothing_classifier.h5"))

# ─── Iconos SVG monocromáticos (stroke-based, 24×24) ─────────────────────────
def _svg(inner: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round">'
        + inner + '</svg>'
    )

ICONS = [
    # 00 Vestido — bodice con falda acampanada
    _svg('<path d="M9.5 3.5Q12 5.5 14.5 3.5L17.5 8.5L15.5 10L17 21H7L8.5 10L6.5 8.5Z"/>'),
    # 01 Sombrero — ala ancha + copa
    _svg('<path d="M12 3V9M7 9Q12 6.5 17 9M3 13.5H21M7 9V13.5M17 9V13.5"/>'),
    # 02 Manga larga — brazos extendidos hasta los bordes
    _svg('<path d="M2 9V15H7.5V21H16.5V15H22V9L18 7C16.5 8 14.5 8.5 12 8.5C9.5 8.5 7.5 8 6 7Z"/>'),
    # 03 Abrigo — largo con solapas
    _svg('<path d="M5.5 4.5L7.5 3L10.5 5.5C11 6 11.5 6.5 12 6.5C12.5 6.5 13 6 13.5 5.5L16.5 3L18.5 4.5L21 9.5L18.5 11V22H5.5V11L3 9.5Z"/><path d="M12 6.5V22M12 11L9 13.5M12 11L15 13.5"/>'),
    # 04 Pantalón — dos piernas con costura central
    _svg('<path d="M6 3H18L17 15L14 22H10L7 15Z"/><line x1="12" y1="3" x2="12" y2="15"/>'),
    # 05 Camisa — cuello en V + línea de botones (punteada)
    _svg('<path d="M4 8.5L8 3.5C9.5 4.5 10.8 5 12 5C13.2 5 14.5 4.5 16 3.5L20 8.5L17.5 10V21H6.5V10Z"/><path d="M12 5V21" stroke-dasharray="1.5 2"/><path d="M10.5 7L12 5L13.5 7"/>'),
    # 06 Zapatos — perfil lateral con lengüeta
    _svg('<path d="M3 17.5C3 15 6 11.5 10 11.5H17C20 11.5 22 13.5 22 16V18H3Z"/><path d="M10 11.5V7.5"/><path d="M7.5 7.5H13.5"/>'),
    # 07 Shorts — pantalón corto con costura central
    _svg('<path d="M6 3H18L16.5 13L14 14.5H10L7.5 13Z"/><line x1="12" y1="3" x2="12" y2="14.5"/>'),
    # 08 Falda — trapecio con cinturilla
    _svg('<path d="M8 3H16L21 21H3Z"/><line x1="7" y1="7.5" x2="17" y2="7.5"/>'),
    # 09 Camiseta — silueta T clásica, cuello redondo, mangas cortas
    _svg('<path d="M3.5 8L8 3.5C9.5 4.5 10.8 5 12 5C13.2 5 14.5 4.5 16 3.5L20.5 8L18 9.5V21H6V9.5Z"/>'),
]

# ─── Estilos CSS — Dark Glassmorphism ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Fuente & reset ─────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ── Fondo global ───────────────────────────────────── */
.stApp,
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #050C1F 0%, #091428 45%, #0D1E46 100%) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Ruido sutil sobre el fondo */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
}

/* Orbs de luz de fondo */
.stApp::after {
    content: "";
    position: fixed;
    top: -20%;
    right: -10%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(circle, rgba(29,78,216,0.18) 0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(5, 12, 31, 0.75) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(99, 160, 255, 0.12) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.45) !important; }

/* ── Header / toolbar ───────────────────────────────── */
[data-testid="stHeader"],
[data-testid="stToolbar"] {
    background: rgba(5,12,31,0.85) !important;
    backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid rgba(99,160,255,0.10) !important;
}
[data-testid="stDecoration"] { display: none !important; }

/* ── Contenedor principal ───────────────────────────── */
.main .block-container {
    padding: 1.2rem 1.5rem 3rem !important;
    max-width: 1200px;
}

/* ── Texto global ───────────────────────────────────── */
.stApp p, .stApp li, .stApp label,
.stApp [data-testid="stText"] {
    color: rgba(255,255,255,0.80) !important;
}
.stApp h1, .stApp h2, .stApp h3 {
    color: rgba(255,255,255,0.95) !important;
}
.stApp code {
    background: rgba(99,160,255,0.15) !important;
    color: #93C5FD !important;
    border-radius: 4px;
    padding: 1px 5px;
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 0.25rem !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    gap: 0.25rem !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: rgba(255,255,255,0.50) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,0.22) !important;
    color: #93C5FD !important;
    border: 1px solid rgba(59,130,246,0.35) !important;
    box-shadow: 0 0 16px rgba(59,130,246,0.15) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Botón principal ─────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1D4ED8, #2563EB) !important;
    border: 1px solid rgba(99,160,255,0.4) !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    box-shadow: 0 0 24px rgba(37,99,235,0.35), 0 4px 12px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 40px rgba(37,99,235,0.55), 0 4px 16px rgba(0,0,0,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Botón secundario (limpiar historial, etc.) */
.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: rgba(255,255,255,0.65) !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.09) !important;
    color: rgba(255,255,255,0.85) !important;
}

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(99,160,255,0.3) !important;
    border-radius: 12px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] * { color: rgba(255,255,255,0.70) !important; }
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary { color: rgba(255,255,255,0.75) !important; }

/* ── Info / Alert ────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(29,78,216,0.12) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.80) !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Scrollbar ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,160,255,0.25); border-radius: 3px; }

/* ══════════════════════════════════════════════════════
   COMPONENTES PERSONALIZADOS
   ══════════════════════════════════════════════════════ */

/* Hero banner */
.hero {
    background: linear-gradient(135deg,
        rgba(29,78,216,0.55) 0%,
        rgba(13,51,140,0.65) 50%,
        rgba(7,29,82,0.70) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(99,160,255,0.22);
    border-radius: 20px;
    padding: 3rem 2rem 2.5rem;
    text-align: center;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 0 80px rgba(29,78,216,0.18),
        0 8px 32px rgba(0,0,0,0.45),
        inset 0 1px 0 rgba(255,255,255,0.10);
}
.hero::before {
    content: "";
    position: absolute;
    top: -40%;
    left: 50%;
    transform: translateX(-50%);
    width: 70%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(59,130,246,0.20) 0%, transparent 60%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #60A5FA;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: rgba(255,255,255,0.97) !important;
    margin: 0;
    letter-spacing: -1.5px;
    line-height: 1.1;
}
.hero h1 span { color: #60A5FA; }
.hero p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.62) !important;
    margin: 0.8rem 0 0;
    max-width: 520px;
    margin-left: auto;
    margin-right: auto;
}
.demo-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(234,88,12,0.20);
    border: 1px solid rgba(251,146,60,0.35);
    color: #FB923C;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 1rem;
}

/* Glass card base */
.glass-card {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.07);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(99,160,255,0.20);
    box-shadow: 0 4px 32px rgba(0,0,0,0.4), 0 0 24px rgba(29,78,216,0.08),
                inset 0 1px 0 rgba(255,255,255,0.09);
}

/* Tarjeta resultado */
.result-box {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(99,160,255,0.20);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    box-shadow:
        0 0 40px rgba(29,78,216,0.12),
        0 8px 24px rgba(0,0,0,0.40),
        inset 0 1px 0 rgba(255,255,255,0.08);
    margin-top: 0.8rem;
    position: relative;
    overflow: hidden;
}
.result-box::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,160,255,0.5), transparent);
}
.result-class {
    font-size: 2rem;
    font-weight: 800;
    color: rgba(255,255,255,0.97) !important;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.result-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #60A5FA !important;
    margin-bottom: 0.35rem;
    display: block;
}
.conf-high { color: #4ADE80 !important; font-weight: 700; font-size: 1.1rem; }
.conf-mid  { color: #FBBF24 !important; font-weight: 700; font-size: 1.1rem; }
.conf-low  { color: #F87171 !important; font-weight: 700; font-size: 1.1rem; }
.meta-row  { color: rgba(255,255,255,0.38) !important; font-size: 0.78rem; margin-top: 0.5rem; }
.meta-row code { color: #93C5FD !important; }

/* Estadísticas */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-top: 1.2rem;
}
.stat {
    background: rgba(255,255,255,0.045);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.1rem 0.8rem;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stat:hover {
    border-color: rgba(99,160,255,0.22);
    box-shadow: 0 0 20px rgba(29,78,216,0.10);
}
.stat-num {
    font-size: 1.75rem;
    font-weight: 800;
    color: #60A5FA !important;
    letter-spacing: -0.5px;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.72rem;
    font-weight: 500;
    color: rgba(255,255,255,0.42) !important;
    margin-top: 0.3rem;
    letter-spacing: 0.5px;
}

/* Info cards (about / tech) */
.info-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    transition: border-color 0.2s, background 0.2s;
}
.info-card:hover {
    background: rgba(255,255,255,0.06);
    border-color: rgba(99,160,255,0.18);
}
.info-card strong { color: rgba(255,255,255,0.90) !important; }
.info-card span   { color: rgba(255,255,255,0.50) !important; }

/* KPI cards */
.kpi-card {
    background: rgba(29,78,216,0.08);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
}
.kpi-card strong { color: rgba(255,255,255,0.75) !important; font-size: 0.85rem; }
.kpi-val { color: #60A5FA !important; font-size: 1.05rem; font-weight: 700; display: block; margin-top: 0.1rem; }

/* Historial de sesión */
.hist-item {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0.7rem;
    border-radius: 8px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    margin: 0.25rem 0;
    font-size: 0.85rem;
}
.hist-item .tag {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 1px;
    color: rgba(255,255,255,0.35) !important;
}

/* Placeholder upload vacío */
.upload-placeholder {
    border: 1px dashed rgba(99,160,255,0.28);
    border-radius: 14px;
    padding: 2.5rem 1rem;
    text-align: center;
    background: rgba(29,78,216,0.06);
    color: rgba(255,255,255,0.40) !important;
}
.upload-placeholder .icon {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}

/* Estado sidebar */
.status-ok   { color: #4ADE80 !important; font-weight: 600; }
.status-demo { color: #FB923C !important; font-weight: 600; }

/* Categorías sidebar */
.cat-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.28rem 0.45rem;
    border-radius: 8px;
    margin: 0.14rem 0;
    transition: background 0.15s;
    cursor: default;
}
.cat-item:hover { background: rgba(255,255,255,0.05); }
.cat-item:hover .cat-icon {
    background: rgba(59,130,246,0.20);
    border-color: rgba(59,130,246,0.40);
    color: #93C5FD;
}
.cat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    min-width: 30px;
    background: rgba(59,130,246,0.10);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 8px;
    color: #60A5FA;
    padding: 4px;
    transition: all 0.15s;
}
.cat-icon svg { width: 100%; height: 100%; }
.cat-num {
    font-size: 0.60rem;
    font-weight: 600;
    color: rgba(255,255,255,0.22) !important;
    font-family: monospace;
    letter-spacing: 0.3px;
    margin-left: auto;
}
.cat-name { font-size: 0.87rem; color: rgba(255,255,255,0.72) !important; flex: 1; }

/* Section header dentro de tabs */
.section-label {
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #3B82F6 !important;
    margin-bottom: 0.6rem;
    margin-top: 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: "";
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* Tech stack items */
.tech-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.65rem 1rem;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    margin: 0.3rem 0;
}
.tech-name { color: rgba(255,255,255,0.82) !important; font-weight: 500; font-size: 0.9rem; }
.tech-ver  { color: #3B82F6 !important; font-size: 0.78rem; font-weight: 600;
             background: rgba(59,130,246,0.12); border-radius: 5px; padding: 1px 7px; }

/* Tabla markdown */
.stApp table {
    border-collapse: collapse;
    width: 100%;
    background: transparent !important;
}
.stApp th {
    background: rgba(29,78,216,0.18) !important;
    color: #93C5FD !important;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid rgba(99,160,255,0.15) !important;
    padding: 0.6rem 0.9rem !important;
}
.stApp td {
    background: rgba(255,255,255,0.03) !important;
    color: rgba(255,255,255,0.72) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    padding: 0.5rem 0.9rem !important;
    font-size: 0.88rem;
}

/* Code block */
.stApp pre {
    background: rgba(0,0,0,0.35) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: rgba(255,255,255,0.22) !important;
    font-size: 0.78rem;
    letter-spacing: 0.3px;
}
.app-footer strong { color: rgba(255,255,255,0.40) !important; }

/* ── Responsive: móvil ───────────────────────────────── */
@media (max-width: 768px) {
    .hero h1            { font-size: 1.9rem !important; letter-spacing: -0.5px; }
    .hero p             { font-size: 0.9rem !important; }
    .hero                { padding: 2rem 1.2rem 1.8rem !important; }

    .main .block-container { padding: 0.8rem 0.8rem 2rem !important; }

    /* Stack todas las columnas en móvil */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"] {
        width: 100% !important;
        flex: 0 0 100% !important;
        min-width: 100% !important;
    }

    /* Stats: 2×2 en móvil */
    .stat-grid { grid-template-columns: repeat(2, 1fr) !important; }

    .result-class   { font-size: 1.6rem !important; }
    .result-box     { padding: 1.2rem 1.2rem !important; }

    /* Tabs más compactos */
    .stTabs [data-baseweb="tab"] {
        font-size: 0.78rem !important;
        padding: 0.45rem 0.7rem !important;
    }
}

@media (max-width: 480px) {
    .hero h1 { font-size: 1.6rem !important; }
    .stat-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .stat-num  { font-size: 1.4rem !important; }
}
</style>
""", unsafe_allow_html=True)


# ─── Carga del modelo (cacheado) ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(str(MODEL_PATH))
    except Exception as exc:
        st.warning(f"Error al cargar el modelo: {exc}")
        return None


# ─── Preprocesamiento ─────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE, Image.BICUBIC)
    x = np.array(img, dtype=np.float32)
    x = (x / 127.5) - 1.0
    return np.expand_dims(x, 0)


# ─── Inferencia ───────────────────────────────────────────────────────────────
def predict_real(model, img: Image.Image):
    x = preprocess(img)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    return probs, time.perf_counter() - t0


def predict_demo(img: Image.Image):
    seed = int(np.array(img.resize((32, 32))).mean() * 10) % 9999
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=0.6, size=10)
    raw[rng.integers(10)] *= 4
    return raw / raw.sum(), 0.038


# ─── Gráfica de confianza ─────────────────────────────────────────────────────
def confidence_chart(probs: np.ndarray) -> go.Figure:
    top_i    = int(np.argmax(probs))
    sorted_i = np.argsort(probs)[::-1][:5]
    labels   = [CLASES[i][0] for i in sorted_i]
    values   = [probs[i] * 100 for i in sorted_i]
    colors   = ["#3B82F6" if i == top_i else "rgba(59,130,246,0.28)" for i in sorted_i]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="rgba(255,255,255,0.65)", size=12),
    ))
    fig.update_layout(
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
            tickfont=dict(color="rgba(255,255,255,0.35)", size=10),
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="rgba(255,255,255,0.72)", size=12),
            autorange="reversed",
        ),
        height=240,
        margin=dict(l=0, r=40, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(
            bgcolor="rgba(15,30,70,0.9)",
            bordercolor="rgba(99,160,255,0.3)",
            font_color="white",
        ),
    )
    return fig


# ─── Sesión ───────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total" not in st.session_state:
    st.session_state.total = 0


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:700;color:rgba(255,255,255,0.92);'>"
        "◈ RapidRelief AI</div>"
        "<div style='font-size:0.78rem;color:rgba(255,255,255,0.38);margin-top:0.2rem;'>"
        "Clasificación textil humanitaria</div>",
        unsafe_allow_html=True
    )
    st.divider()

    model = load_model()

    if model is not None:
        st.markdown('<p class="status-ok">◉ Modelo activo</p>', unsafe_allow_html=True)
        st.caption(f"`{MODEL_PATH}`")
    else:
        st.markdown('<p class="status-demo">◎ Modo demo</p>', unsafe_allow_html=True)
        st.caption("Predicciones simuladas · sin modelo cargado")
        with st.expander("¿Cómo cargar el modelo?"):
            st.markdown(f"""
1. Entrena en `02_transfer_learning.ipynb`
2. Descarga el `.h5` desde Colab
3. Colócalo en:
```
{MODEL_PATH}
```
4. Reinicia la app
""")

    st.divider()
    st.markdown(
        "<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;"
        "text-transform:uppercase;color:rgba(255,255,255,0.30);margin-bottom:0.5rem;'>"
        "Categorías</div>",
        unsafe_allow_html=True
    )
    for (es, en, num), icon in zip(CLASES, ICONS):
        st.markdown(
            f'<div class="cat-item">'
            f'<span class="cat-icon">{icon}</span>'
            f'<span class="cat-name">{es}</span>'
            f'<span class="cat-num">{num}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.divider()

    if st.session_state.total > 0:
        st.markdown(
            f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:2px;"
            f"text-transform:uppercase;color:rgba(255,255,255,0.30);margin-bottom:0.5rem;'>"
            f"Sesión · {st.session_state.total} imagen(es)</div>",
            unsafe_allow_html=True
        )
        for es, conf, is_demo in reversed(st.session_state.history[-5:]):
            tag = "DEMO" if is_demo else "REAL"
            st.markdown(
                f'<div class="hist-item">'
                f'<span style="color:rgba(255,255,255,0.82);flex:1;">{es}</span>'
                f'<span style="color:#60A5FA;font-weight:600;">{conf:.0%}</span>'
                f'<span class="tag">{tag}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        if st.button("Limpiar historial", use_container_width=True):
            st.session_state.history = []
            st.session_state.total = 0
            st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:0.78rem;color:rgba(255,255,255,0.35);line-height:1.7;'>"
        "Fernando Acuña · Pamela Ruíz<br>Dijo Lozada<br>"
        "<span style='color:rgba(255,255,255,0.22);'>Dr. José A. Bastián · USB 2026</span>"
        "</div>",
        unsafe_allow_html=True
    )


# ─── Hero ─────────────────────────────────────────────────────────────────────
demo_badge = (
    '<div><span class="demo-badge">◎ &nbsp;MODO DEMO — predicciones simuladas</span></div>'
    if model is None else ""
)
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">◈ &nbsp;VISIÓN POR COMPUTADORA &nbsp;·&nbsp; RESPUESTA HUMANITARIA</div>
  <h1>Rapid<span>Relief</span> AI</h1>
  <p>Clasificación automatizada de donaciones textiles para respuesta humanitaria en campo</p>
  {demo_badge}
</div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_cls, tab_proj, tab_tech = st.tabs(["◈ Clasificar", "◆ El Proyecto", "◉ Detalles Técnicos"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASIFICAR
# ══════════════════════════════════════════════════════════════════════════════
with tab_cls:
    col_up, col_res = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<div class="section-label">Entrada</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Arrastra o selecciona una imagen de prenda",
            type=["jpg", "jpeg", "png", "webp"],
            help="Formatos: JPG · PNG · WEBP · Máx. 10 MB",
        )

        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption=uploaded.name, use_container_width=True)
            do_classify = st.button("◈ &nbsp;Clasificar imagen", type="primary", use_container_width=True)
        else:
            st.markdown("""
<div class="upload-placeholder">
  <div class="icon">◈</div>
  <div style="font-size:0.95rem;font-weight:500;color:rgba(255,255,255,0.45);">
    Sube una imagen para clasificar
  </div>
  <div style="font-size:0.78rem;margin-top:0.3rem;color:rgba(255,255,255,0.22);">
    JPG · PNG · WEBP · Máx. 10 MB
  </div>
</div>
""", unsafe_allow_html=True)
            do_classify = False

    with col_res:
        st.markdown('<div class="section-label">Resultado</div>', unsafe_allow_html=True)

        if uploaded and do_classify:
            with st.spinner("Analizando…"):
                if model is not None:
                    probs, elapsed = predict_real(model, img)
                    is_demo = False
                else:
                    probs, elapsed = predict_demo(img)
                    is_demo = True

            top_i    = int(np.argmax(probs))
            top_prob = float(probs[top_i])
            conf_cls = "conf-high" if top_prob > 0.7 else ("conf-mid" if top_prob > 0.4 else "conf-low")
            demo_note = (
                '<span style="background:rgba(234,88,12,0.18);border:1px solid rgba(251,146,60,0.30);'
                'color:#FB923C;border-radius:6px;padding:0.1rem 0.55rem;font-size:0.68rem;'
                'font-weight:700;letter-spacing:1px;">DEMO</span> '
            ) if is_demo else ""

            st.markdown(f"""
<div class="result-box">
  {demo_note}
  <span class="result-label">Categoría detectada</span>
  <div class="result-class">{CLASES[top_i][0]}</div>
  <div class="{conf_cls}" style="margin-top:0.25rem;">{top_prob:.1%} confianza</div>
  <div class="meta-row">
    ◈ &nbsp;{elapsed*1000:.0f} ms &nbsp;·&nbsp;
    índice [{top_i}] &nbsp;·&nbsp; <code>{CLASES[top_i][1]}</code>
  </div>
</div>
""", unsafe_allow_html=True)

            st.plotly_chart(confidence_chart(probs), use_container_width=True)

            st.session_state.history.append((CLASES[top_i][0], top_prob, is_demo))
            st.session_state.total += 1

            if is_demo:
                st.info(
                    "Resultado simulado. Coloca el modelo entrenado en "
                    f"`{MODEL_PATH}` y reinicia la app para predicciones reales.",
                    icon="◎"
                )

        elif not uploaded:
            st.markdown("""
<div style="padding:3.5rem 1rem;text-align:center;color:rgba(255,255,255,0.20);">
  <div style="font-size:2.5rem;font-weight:300;letter-spacing:4px;">◈</div>
  <div style="font-size:0.9rem;margin-top:0.5rem;">
    Sube una imagen y presiona <strong style="color:rgba(255,255,255,0.40);">Clasificar</strong>
  </div>
</div>
""", unsafe_allow_html=True)

    # Stats
    st.markdown("""
<div class="stat-grid">
  <div class="stat"><div class="stat-num">10</div><div class="stat-lbl">Categorías</div></div>
  <div class="stat"><div class="stat-num">≥ 90%</div><div class="stat-lbl">Precisión objetivo</div></div>
  <div class="stat"><div class="stat-num">&lt; 300ms</div><div class="stat-lbl">Latencia</div></div>
  <div class="stat"><div class="stat-num">Offline</div><div class="stat-lbl">Modo final</div></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EL PROYECTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_proj:
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<div class="section-label">El Problema</div>', unsafe_allow_html=True)
        st.markdown("""
Durante desastres naturales y crisis humanitarias los centros de donación reciben
**grandes volúmenes de ropa** que deben clasificarse manualmente. Este proceso es
lento, agotador y propenso a errores — limitando la velocidad de distribución de ayuda.

> *"El tiempo de clasificación puede ser la diferencia entre recibir ayuda a tiempo o no."*
""")

        st.markdown('<div class="section-label">La Solución</div>', unsafe_allow_html=True)
        st.markdown("""
**RapidRelief AI** automatiza la clasificación con visión por computadora:

- Fotografía la prenda → MobileNetV2 la clasifica en **< 300 ms**
- Guía la distribución hacia quienes más lo necesitan
- Funciona **100% offline** — sin internet en campo
- Reduce el tiempo de clasificación manual en **70%**
""")

        st.markdown('<div class="section-label">Datos de Entrenamiento</div>', unsafe_allow_html=True)
        st.markdown("""
| Dataset | Imágenes | Tipo |
|---------|----------|------|
| Clothing Dataset Small | ~5 000 | Fotografías reales en color |
| Fashion-MNIST (Zalando) | 70 000 | Escala de grises 28×28 |

División: **85% entrenamiento / 15% prueba** · Sin datos biométricos.
""")

    with col_r:
        st.markdown('<div class="section-label">KPIs del Proyecto</div>', unsafe_allow_html=True)
        kpis = [
            ("Precisión (val_accuracy)",     "≥ 90%"),
            ("F1-Score por clase",            "≥ 0.88"),
            ("Latencia de inferencia",        "≤ 300 ms"),
            ("Tamaño modelo TFLite",          "≤ 6 MB"),
            ("Funcionalidad offline",         "100%"),
            ("Reducción vs. clasificación manual", "70%"),
        ]
        for k, v in kpis:
            st.markdown(
                f'<div class="kpi-card"><strong>{k}</strong>'
                f'<span class="kpi-val">{v}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-label">Equipo</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="info-card">
  <div style="color:rgba(255,255,255,0.85);font-weight:600;font-size:0.9rem;">
    Fernando Acuña Martínez<br>
    Pamela Ruíz Velasco Calvo<br>
    Dijo Lozada Vivar
  </div>
  <div style="color:rgba(255,255,255,0.35);font-size:0.78rem;margin-top:0.5rem;">
    Dr. José Ambrosio Bastián · USB 2026
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DETALLES TÉCNICOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<div class="section-label">Arquitectura</div>', unsafe_allow_html=True)
        st.markdown("""
**Base:** MobileNetV2 pre-entrenado en ImageNet
3.5 M parámetros · 14 MB (.h5) → ~4 MB (.tflite INT8)

**Head personalizado:**
```
Input (224 × 224 × 3)
└── MobileNetV2  [congelada en Fase A]
    └── GlobalAveragePooling2D
        └── Dense(2048, relu)
            └── Dense(512, relu)
                └── Dense(16, sigmoid)
                    └── Dense(10, softmax)
```

| Fase | Descripción | Épocas | LR |
|------|-------------|--------|----|
| A | Feature extraction · base congelada | 10 | 1e-3 |
| B | Fine-tuning · últimas 54 capas | 10–15 | 1e-5 |
""")

        st.markdown('<div class="section-label">Pipeline de Datos</div>', unsafe_allow_html=True)
        st.markdown("""
```python
ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=10.0,
    zoom_range=0.1,
    horizontal_flip=True,
)
# Entrada: 224×224 · rango [-1, 1]
```
Fashion-MNIST: grayscale 28×28 → RGB stack ×3 → resize 224×224
""")

    with col_r:
        st.markdown('<div class="section-label">Stack Tecnológico</div>', unsafe_allow_html=True)
        tech = [
            ("Python",            "3.13"),
            ("TensorFlow",        "2.21"),
            ("Keras",             "3.x"),
            ("NumPy",             "2.x"),
            ("Pillow",            "11.x"),
            ("Streamlit",         "1.56 · esta app"),
            ("Flutter + TFLite",  "Deploy final"),
        ]
        for name, ver in tech:
            st.markdown(
                f'<div class="tech-row">'
                f'<span class="tech-name">{name}</span>'
                f'<span class="tech-ver">{ver}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-label">Referencia de Rendimiento</div>', unsafe_allow_html=True)
        st.markdown("""
Benchmark del notebook de referencia (Xception, Clothing Dataset Small):

| Config | Val Acc |
|--------|---------|
| LR=0.01, sin dropout | 82.7% |
| LR=0.001, dropout=0.2 | 84.8% |
| Con augmentación | 84.5% |
| Xception 299×299 | **88.3%** |

MobileNetV2 + fine-tuning apunta a superar ese baseline.
""")

        st.markdown('<div class="section-label">Flujo de Deployment</div>', unsafe_allow_html=True)
        st.markdown("""
```
Colab (GPU T4) → model.h5
  → TFLiteConverter + INT8
  → model.tflite (~4 MB)
  → Flutter app (offline)
```
Esta app web: Streamlit Cloud · validación durante desarrollo.
""")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  <strong>◈ RapidRelief AI</strong> &nbsp;·&nbsp;
  Clasificación automatizada de donaciones textiles &nbsp;·&nbsp; USB 2026<br>
  MobileNetV2 · Transfer Learning · TensorFlow 2.21 · Streamlit 1.56
</div>
""", unsafe_allow_html=True)
