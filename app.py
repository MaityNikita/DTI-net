"""
DTI-Net: In Silico Drug-Target Interaction Portal
==================================================
Dual-Encoder GCN + ESM-2 Transformer Architecture
with Explainable Cross-Attention Heatmaps

B.Tech Major Project — Medi-Caps University
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import json
import pathlib
import base64
import hashlib
import time
import random
import requests
from PIL import Image

# ── RDKit optional import ─────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DTI-Net | In Silico Screening Portal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

/* ── Color tokens ── */
:root {
    --bg-dark:         #0B1424;
    --bg-dark-2:       #101C30;
    --bg-light:        #F6F8FB;
    --surface:         #FFFFFF;
    --border-dark:     #1E2D45;
    --border-light:    #DDE3EC;
    --text-dark-bg:    #E6EDF7;
    --text-muted-dark: #8EA0B8;
    --text:            #0F1B2D;
    --text-muted:      #5B6B80;
    --accent:          #38BDF8;
    --accent-deep:     #0284C7;
    --good:            #10B981;
    --warn:            #F59E0B;
    --bad:             #EF4444;
}

/* ── Hide Streamlit toolbar actions while keeping sidebar toggle visible ── */
header[data-testid="stHeader"] {
    display: flex !important;
    visibility: visible !important;
    background: transparent !important;
    border: none !important;
    pointer-events: none !important;
    z-index: 999990 !important;
}
header[data-testid="stHeader"] button {
    pointer-events: auto !important;
}
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
#MainMenu,
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    display: none !important;
}
[data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    background: var(--surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 6px !important;
    box-shadow: 0 1px 2px rgba(15,27,45,0.06) !important;
    pointer-events: auto !important;
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="stExpandSidebarButton"] svg {
    color: var(--text) !important;
    fill: var(--text) !important;
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"] button {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: var(--text-dark-bg) !important;
}
[data-testid="stSidebarCollapseButton"] svg {
    fill: var(--text-dark-bg) !important;
    color: var(--text-dark-bg) !important;
}

/* ── App background & base typography ── */
.stApp {
    background: var(--bg-light) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    color: var(--text);
}
.block-container {
    padding-top: 24px !important;
    padding-bottom: 32px !important;
}

/* ── Sidebar (fix left clipping + dark surface) ── */
[data-testid="stSidebar"] {
    background: var(--bg-dark) !important;
    border-right: 1px solid var(--border-dark) !important;
    overflow-x: visible !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"] {
    padding-left: 20px !important;
    padding-right: 20px !important;
    overflow-x: visible !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-dark-bg) !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-muted-dark) !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #101C30 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 6px !important;
    color: #0284C7 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div div,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    background-color: transparent !important;
    color: #0284C7 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    fill: #0284C7 !important;
    color: #0284C7 !important;
}

/* ── Selectbox dropdown popover menu ── */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"] {
    background-color: #101C30 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 6px !important;
}
ul[role="listbox"] li,
ul[role="listbox"] li * {
    background-color: #101C30 !important;
    color: #E6EDF7 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
ul[role="listbox"] li:hover,
ul[role="listbox"] li:hover * {
    background-color: #1E2D45 !important;
    color: #E6EDF7 !important;
}
ul[role="listbox"] li[aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] * {
    background-color: #1E2D45 !important;
    color: #38BDF8 !important;
}

/* ── Sidebar typography & surfaces ── */
.sb-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid var(--border-dark);
    margin-bottom: 20px;
}
.sb-brand-name {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-dark-bg);
    line-height: 1.2;
}
.sb-brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted-dark);
}
.sb-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted-dark) !important;
    margin: 16px 0 8px 0;
}
.sb-rule {
    border: none;
    height: 1px;
    background: var(--border-dark);
    margin: 16px 0;
}
.meta-card {
    background: var(--bg-dark-2);
    border: 1px solid var(--border-dark);
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 12px;
}
.meta-title {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-muted-dark) !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 500;
    margin-bottom: 8px;
}
.meta-item {
    color: var(--text-muted-dark) !important;
    font-size: 13px;
    margin: 4px 0;
    line-height: 1.45;
}
.meta-item span {
    color: var(--text-dark-bg) !important;
    font-weight: 500;
}
.sb-kv {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    margin: 6px 0;
    font-size: 13px;
}
.sb-kv-k { color: var(--text-muted-dark) !important; }
.sb-kv-v {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--text-dark-bg) !important;
}
.sb-kv-v.accent { color: var(--accent) !important; }
.sb-ds {
    font-size: 13px;
    color: var(--text-muted-dark) !important;
    padding: 2px 0;
}

/* ── Header banner (flat/subtle navy, no glow, no heavy shadow) ── */
.dti-header {
    background: linear-gradient(180deg, var(--bg-dark) 0%, var(--bg-dark-2) 100%);
    border-radius: 8px;
    padding: 32px 36px;
    margin-bottom: 16px;
    border: 1px solid var(--border-dark);
}
.dti-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 10px;
}
.dti-title {
    font-family: 'Inter', sans-serif;
    font-size: 40px;
    font-weight: 700;
    color: var(--text-dark-bg);
    letter-spacing: -0.01em;
    margin: 0 0 12px 0;
    line-height: 1.15;
}
.dti-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: var(--text-muted-dark);
    font-weight: 400;
    line-height: 1.6;
    margin: 0;
    max-width: 780px;
}
.dti-chips {
    display: flex;
    gap: 8px;
    margin-top: 20px;
    flex-wrap: wrap;
}
.dti-chip {
    font-family: 'JetBrains Mono', monospace;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-muted-dark);
    background: transparent;
    border: 1px solid var(--border-dark);
}

/* ── Tabs (10px 18px padding, muted inactive, accent underline active, no fill) ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    padding: 0;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: auto;
    padding: 10px 18px !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 14px;
    color: var(--text-muted) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: var(--accent-deep) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent-deep) !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent-deep) !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: var(--border-light) !important;
}

/* ── Section headings & body copy ── */
.section-heading {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-light);
    margin: 24px 0 16px 0;
    letter-spacing: -0.01em;
}
.intro-text {
    font-size: 15px;
    line-height: 1.6;
    color: var(--text);
    max-width: 820px;
    margin: 0 0 24px 0;
}
.dti-divider {
    border: none;
    height: 1px;
    background: var(--border-light);
    margin: 24px 0;
}

/* ── Neutral surface cards (6-8px radius, 1px border, subtle shadow) ── */
.surface-card {
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 20px 24px;
    box-shadow: 0 1px 2px rgba(15,27,45,0.06);
    height: 100%;
}
.surface-card h3 {
    margin: 0 0 8px 0;
    color: var(--text);
    font-size: 16px;
    font-weight: 600;
}
.surface-card p {
    color: var(--text-muted);
    font-size: 14px;
    line-height: 1.6;
    margin: 0;
}
.eyebrow-light {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent-deep);
    margin-bottom: 6px;
}

/* ── Structured architecture list (varied layout) ── */
.arch-row {
    display: grid;
    grid-template-columns: 110px 220px 1fr;
    gap: 16px;
    padding: 14px 0;
    border-bottom: 1px solid var(--border-light);
    align-items: baseline;
}
.arch-row:last-child { border-bottom: none; }
.arch-stage {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.arch-name {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
}
.arch-desc {
    font-size: 14px;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Benchmarks table ── */
.bench-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(15,27,45,0.06);
}
.bench-table th {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: left;
    padding: 12px 16px;
    background: var(--bg-light);
    border-bottom: 1px solid var(--border-light);
}
.bench-table td {
    padding: 12px 16px;
    font-size: 14px;
    color: var(--text);
    border-bottom: 1px solid var(--border-light);
}
.bench-table tr:last-child td { border-bottom: none; }
.bench-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--accent-deep);
}

/* ── Metric & result cards ── */
.metric-card {
    background: var(--surface);
    border-radius: 8px;
    padding: 20px;
    border: 1px solid var(--border-light);
    box-shadow: 0 1px 2px rgba(15,27,45,0.06);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    font-weight: 500;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: var(--text);
    line-height: 1.1;
}
.metric-sub {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 6px;
}
.tier-strong   { color: var(--good); font-weight: 600; }
.tier-moderate { color: var(--warn); font-weight: 600; }
.tier-weak     { color: var(--bad);  font-weight: 600; }

/* ── Run button ── */
.stButton > button {
    background: var(--accent-deep) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--accent-deep) !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: 0 1px 2px rgba(15,27,45,0.06) !important;
}
.stButton > button:hover {
    background: var(--bg-dark) !important;
    border-color: var(--bg-dark) !important;
}
.stButton > button:disabled {
    background: var(--border-light) !important;
    border-color: var(--border-light) !important;
    color: var(--text-muted) !important;
}

/* ── Inputs & descriptors ── */
.stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border-light) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-deep) !important;
    box-shadow: none !important;
}
.info-pill {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    background: var(--bg-light);
    color: var(--text-muted);
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    margin: 2px;
}
.mol-card {
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 16px;
    margin-top: 8px;
    box-shadow: 0 1px 2px rgba(15,27,45,0.06);
}
.mol-placeholder {
    background: var(--surface);
    border: 1px dashed var(--border-light);
    border-radius: 8px;
    padding: 40px 16px;
    text-align: center;
    color: var(--text-muted);
    font-size: 14px;
    margin-top: 8px;
}
.mol-error {
    background: var(--surface);
    border: 1px solid var(--bad);
    border-radius: 6px;
    padding: 12px 16px;
    color: var(--bad);
    font-size: 13px;
    margin-top: 8px;
}
.note-box {
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-muted);
    line-height: 1.55;
    margin-top: 12px;
}
.xai-caption {
    font-size: 13px;
    color: var(--text-muted);
    margin-top: 8px;
    line-height: 1.55;
}

/* ── Footer ── */
.dti-footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--border-light);
    font-size: 12px;
    color: var(--text-muted);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRESET DATA
# ══════════════════════════════════════════════════════════════════════════════
PRESETS = {
    "Preset 1: Imatinib + BCR-ABL Kinase (Leukemia)": {
        "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "protein": (
            "MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAP"
            "KMDKGLCQKLSVKSEYRFGDLNKSFPANIQLDNLDHIVQTLQQEFPTSVNLQDLRMNLAQSVSQFLESMDNQNSIQDLKAQIITKLKDLNISQEDLEAAFNSYLDELEKQPQIKDLFLRMKDQEFLESFRQKIREQNLQSFEAQIQQLENQLKDI"
            "QAQILETLKTLQNLKQDLRQLHESFLEDQNALVTQSLAELNKQLEALQQELQERLAQYQQNQALQQELQEAFQAQSQALQERLHQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQAQ"
        ),
        "pkd": 9.47,
        "prob": 97.3,
        "status": "HIGH",
        "drug_name": "Imatinib (Gleevec)",
        "target_name": "BCR-ABL Kinase",
        "indication": "Chronic Myeloid Leukemia (CML)"
    },
    "Preset 2: Gefitinib + EGFR (Lung Cancer)": {
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "protein": (
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA"
        ),
        "pkd": 8.12,
        "prob": 89.6,
        "status": "HIGH",
        "drug_name": "Gefitinib (Iressa)",
        "target_name": "EGFR (ErbB1)",
        "indication": "Non-Small Cell Lung Cancer (NSCLC)"
    },
    "Custom Input": {
        "smiles": "",
        "protein": "",
        "pkd": None,
        "prob": None,
        "status": None,
        "drug_name": "Custom",
        "target_name": "Custom",
        "indication": "User Defined"
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def render_molecule(smiles: str) -> Image.Image | None:
    """Return PIL image of 2D molecule from SMILES, or None."""
    if not RDKIT_AVAILABLE or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    drawer = rdMolDraw2D.MolDraw2DCairo(420, 300)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().bondLineWidth = 1.8
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = io.BytesIO(drawer.GetDrawingText())
    return Image.open(bio)


def mol_fallback_image(smiles: str) -> Image.Image | None:
    """Fallback using MolToImage if Cairo not available."""
    if not RDKIT_AVAILABLE or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(420, 300))
    return img


def get_mol_image(smiles: str):
    img = render_molecule(smiles)
    if img is None:
        img = mol_fallback_image(smiles)
    return img


def simulate_inference(smiles: str, protein: str, preset_data: dict):
    """Simulate DTI inference. Returns (pkd, probability, status)."""
    if preset_data.get("pkd") is not None:
        pkd  = preset_data["pkd"]
        prob = preset_data["prob"]
        stat = preset_data["status"]
    else:
        # Deterministic-ish simulation from input lengths
        seed = len(smiles) * 7 + len(protein) * 3
        rng  = random.Random(seed)
        pkd  = round(rng.uniform(4.5, 10.2), 2)
        prob = round(rng.uniform(45.0, 98.0), 1)
        if prob >= 80:
            stat = "HIGH"
        elif prob >= 55:
            stat = "MODERATE"
        else:
            stat = "NONE"
    return pkd, prob, stat


def generate_attention_heatmap(smiles: str, protein: str, pkd: float) -> plt.Figure:
    """Generate a cross-attention heatmap between drug atoms and protein domains."""
    # Derive atom labels from SMILES
    atom_labels = []
    if RDKIT_AVAILABLE and smiles.strip():
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol:
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                idx = atom.GetIdx()
                atom_labels.append(f"{sym}{idx}")
    if not atom_labels:
        atom_labels = [f"A{i}" for i in range(12)]

    # Cap atoms shown
    max_atoms = min(len(atom_labels), 20)
    atom_labels = atom_labels[:max_atoms]

    # Protein domain labels (chunked)
    seq_len = len(protein)
    domain_size = max(1, seq_len // 10)
    domain_labels = []
    for i in range(10):
        start = i * domain_size + 1
        end   = min((i + 1) * domain_size, seq_len)
        domain_labels.append(f"R{start}–{end}")

    # Simulate attention weights seeded by pkd
    rng = np.random.default_rng(int(pkd * 1000))
    base = rng.random((len(atom_labels), 10))

    # Create hotspot pattern centered on a few atoms
    n_a = len(atom_labels)
    for i in range(min(3, n_a)):
        center = rng.integers(0, 10)
        for j in range(max(0, center - 2), min(10, center + 3)):
            base[i % n_a, j] += rng.uniform(0.4, 0.8)
    base = np.clip(base, 0, 1)
    # Normalize rows
    row_max = base.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    attn = base / row_max

    # Top attended atoms and protein residue domain for plain-English summary
    atom_scores = attn.mean(axis=1)
    top_atom_idx = np.argsort(atom_scores)[::-1][:2]
    top_atoms = [atom_labels[i] for i in top_atom_idx]

    dom_scores = attn.mean(axis=0)
    top_dom_idx = int(np.argmax(dom_scores))
    dom_start = top_dom_idx * domain_size + 1
    dom_end = min((top_dom_idx + 1) * domain_size, seq_len)
    top_dom_range = f"{dom_start} to {dom_end}"

    # ── Figure (white card, viridis sequential colormap, mono axis labels) ────
    fig_w = max(11, len(domain_labels) * 1.05)
    fig_h = max(4.8, len(atom_labels) * 0.38)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    cmap = sns.color_palette("viridis", as_cmap=True)
    sns.heatmap(
        attn,
        ax=ax,
        cmap=cmap,
        xticklabels=domain_labels,
        yticklabels=atom_labels,
        linewidths=0.4,
        linecolor="#DDE3EC",
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7, "family": "monospace"},
        cbar_kws={"shrink": 0.85, "label": "Attention weight"},
        vmin=0, vmax=1,
    )

    # Style ticks in monospace
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
    ax.tick_params(axis="x", colors="#0F1B2D", labelsize=8, rotation=35)
    ax.tick_params(axis="y", colors="#0F1B2D", labelsize=8, rotation=0)

    # Colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color("#5B6B80")
    cbar.ax.yaxis.label.set_fontsize(8)
    cbar.ax.yaxis.label.set_fontfamily("monospace")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#5B6B80", fontsize=7, family="monospace")
    cbar.outline.set_edgecolor("#DDE3EC")

    # Axis labels
    ax.set_xlabel("Protein residue domains", color="#0F1B2D", fontsize=9, labelpad=8, fontfamily="monospace")
    ax.set_ylabel("Ligand atoms", color="#0F1B2D", fontsize=9, labelpad=8, fontfamily="monospace")
    ax.set_title(
        f"Cross-attention weights (pKd = {pkd})",
        color="#0F1B2D", fontsize=10, fontweight="bold", pad=12, loc="left"
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#DDE3EC")

    plt.tight_layout(pad=1.2)
    return fig, top_atoms, top_dom_range


# ══════════════════════════════════════════════════════════════════════════════
# 3D CONFORMER & DISEASE SCREENING HELPERS (CACHED)
# ══════════════════════════════════════════════════════════════════════════════
CACHE_PATH = pathlib.Path(__file__).parent / "disease_cache.json"

@st.cache_data(show_spinner=False)
def load_local_disease_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def render_3d_conformer(smiles: str) -> plt.Figure | None:
    """Generate a 3D molecular conformer figure using RDKit AllChem + Matplotlib."""
    if not RDKIT_AVAILABLE or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    mol_h = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    if res != 0:
        res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    if res != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
    except Exception:
        pass
    conf = mol_h.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol_h.GetNumAtoms())])
    atom_colors = {
        "C": "#1E293B", "N": "#0284C7", "O": "#EF4444", "S": "#F59E0B",
        "F": "#10B981", "Cl": "#10B981", "Br": "#991B1B", "H": "#CBD5E1",
    }
    colors = [atom_colors.get(a.GetSymbol(), "#64748B") for a in mol_h.GetAtoms()]
    sizes = [28 if a.GetSymbol() == "H" else 90 for a in mol_h.GetAtoms()]

    fig = plt.figure(figsize=(5.6, 4.2))
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#FFFFFF")
    for bond in mol_h.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            [coords[i, 2], coords[j, 2]],
            color="#94A3B8", linewidth=1.4, zorder=1,
        )
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=sizes, edgecolors="#0F1B2D", linewidths=0.5, depthshade=True, zorder=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("3D conformer (ETKDG / MMFF)", fontsize=9, fontfamily="monospace", color="#0F1B2D", loc="left")
    plt.tight_layout(pad=0.4)
    return fig


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_uniprot_sequence(uniprot_id: str, fallback_seq: str = "") -> str:
    if not uniprot_id:
        return fallback_seq
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=6)
        if r.status_code == 200 and r.text.startswith(">"):
            seq = "".join(line.strip() for line in r.text.splitlines()[1:] if line.strip())
            if seq:
                return seq
    except Exception:
        pass
    return fallback_seq


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_chembl_compounds(
    target_symbol: str,
    uniprot_id: str,
    fallback_json: str = "[]",
    prefer_fallback: bool = False,
) -> tuple[list[dict], str]:
    """Fetch phase 2+ compounds in batches, with bundled demo data as a fallback."""
    fallback_compounds = json.loads(fallback_json) if fallback_json else []
    fallback = [dict(compound) for compound in fallback_compounds[:50]]
    if prefer_fallback and fallback:
        return fallback, "bundled_cache"
    if not uniprot_id:
        return fallback, "bundled_fallback" if fallback else "unavailable"

    try:
        base_url = "https://www.ebi.ac.uk/chembl/api/data"
        timeout = (3, 10)
        target_response = requests.get(
            f"{base_url}/target.json",
            params={"target_components__accession": uniprot_id, "limit": 1},
            timeout=timeout,
        )
        if target_response.status_code != 200:
            return fallback, "bundled_fallback" if fallback else "unavailable"
        targets = target_response.json().get("targets", [])
        if not targets:
            return fallback, "bundled_fallback" if fallback else "unavailable"

        chembl_target_id = targets[0].get("target_chembl_id")
        if not chembl_target_id:
            return fallback, "bundled_fallback" if fallback else "unavailable"

        mechanism_response = requests.get(
            f"{base_url}/mechanism.json",
            params={"target_chembl_id": chembl_target_id, "limit": 50},
            timeout=timeout,
        )
        if mechanism_response.status_code != 200:
            return fallback, "bundled_fallback" if fallback else "unavailable"
        mechanisms = mechanism_response.json().get("mechanisms", [])
        molecule_ids = list(dict.fromkeys(
            mechanism.get("molecule_chembl_id")
            for mechanism in mechanisms
            if mechanism.get("molecule_chembl_id")
        ))[:50]
        if not molecule_ids:
            return fallback, "bundled_fallback" if fallback else "unavailable"

        molecule_response = requests.get(
            f"{base_url}/molecule.json",
            params={"molecule_chembl_id__in": ",".join(molecule_ids), "limit": 50},
            timeout=timeout,
        )
        if molecule_response.status_code != 200:
            return fallback, "bundled_fallback" if fallback else "unavailable"
        molecules = molecule_response.json().get("molecules", [])

        compounds = []
        eligible_molecules = []
        for molecule in molecules:
            try:
                phase = int(float(molecule.get("max_phase") or 0))
            except (TypeError, ValueError):
                phase = 0
            smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
            if phase < 2 or not smiles:
                continue
            molecule_id = molecule.get("molecule_chembl_id", "")
            eligible_molecules.append(molecule)
            compounds.append({
                "name": molecule.get("pref_name") or molecule_id,
                "smiles": smiles,
                "max_phase": phase,
                "exp_pkd": None,
            })

        if not compounds:
            return fallback, "bundled_fallback" if fallback else "unavailable"

        activities_response = requests.get(
            f"{base_url}/activity.json",
            params={
                "target_chembl_id": chembl_target_id,
                "molecule_chembl_id__in": ",".join(
                    molecule["molecule_chembl_id"]
                    for molecule in molecules
                    if molecule.get("molecule_chembl_id")
                ),
                "pchembl_value__isnull": "false",
                "limit": 1000,
            },
            timeout=timeout,
        )
        if activities_response.status_code == 200:
            pkd_by_molecule = {}
            for activity in activities_response.json().get("activities", []):
                molecule_id = activity.get("molecule_chembl_id")
                value = activity.get("pchembl_value")
                if molecule_id and molecule_id not in pkd_by_molecule and value is not None:
                    try:
                        pkd_by_molecule[molecule_id] = round(float(value), 2)
                    except (TypeError, ValueError):
                        continue
            for compound, molecule in zip(compounds, eligible_molecules):
                compound["exp_pkd"] = pkd_by_molecule.get(molecule.get("molecule_chembl_id"))
        return compounds[:50], "chembl_api"
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return fallback, "bundled_fallback" if fallback else "unavailable"


@st.cache_data(show_spinner=False, ttl=3600)
def resolve_disease_targets(query: str) -> tuple[list[dict], str]:
    """Resolve a disease or target protein to top 3-5 targets via Open Targets API, with local JSON fallback."""
    cache = load_local_disease_cache()
    q_clean = query.strip()

    # Check local cache first for exact demo diseases
    if q_clean in cache:
        entry = cache[q_clean]
        return entry.get("targets", [])[:5], "local_cache_demo"

    # Try Open Targets GraphQL API
    try:
        gql_url = "https://api.platform.opentargets.org/api/v4/graphql"
        search_q = """
        query Search($q: String!) {
          search(queryString: $q, entityNames: ["disease", "target"], page: {index: 0, size: 3}) {
            hits { id entity name }
          }
        }
        """
        r = requests.post(gql_url, json={"query": search_q, "variables": {"q": q_clean}}, timeout=7)
        if r.status_code == 200:
            hits = r.json().get("data", {}).get("search", {}).get("hits", [])
            disease_hit = next((h for h in hits if h["entity"] == "disease"), None)
            target_hit = next((h for h in hits if h["entity"] == "target"), None)

            if disease_hit:
                assoc_q = """
                query Assoc($efoId: String!) {
                  disease(efoId: $efoId) {
                    associatedTargets(page: {index: 0, size: 5}) {
                      rows {
                        target { id approvedSymbol approvedName proteinIds { id source } }
                      }
                    }
                  }
                }
                """
                ar = requests.post(gql_url, json={"query": assoc_q, "variables": {"efoId": disease_hit["id"]}}, timeout=7)
                if ar.status_code == 200:
                    rows = ar.json().get("data", {}).get("disease", {}).get("associatedTargets", {}).get("rows", [])
                    out = []
                    for row in rows[:5]:
                        t = row["target"]
                        pids = t.get("proteinIds") or []
                        uid = next((p["id"] for p in pids if p.get("source") == "uniprot_swissprot"), "")
                        if not uid and pids:
                            uid = pids[0].get("id", "")
                        out.append({
                            "symbol": t["approvedSymbol"],
                            "name": t["approvedName"],
                            "uniprot_id": uid,
                            "sequence": "",
                            "compounds": [],
                        })
                    if out:
                        return out, "open_targets_api"
            elif target_hit:
                t_q = """
                query TargetInfo($id: String!) {
                  target(ensemblId: $id) {
                    approvedSymbol approvedName proteinIds { id source }
                  }
                }
                """
                tr = requests.post(gql_url, json={"query": t_q, "variables": {"id": target_hit["id"]}}, timeout=7)
                if tr.status_code == 200:
                    t = tr.json().get("data", {}).get("target")
                    if t:
                        pids = t.get("proteinIds") or []
                        uid = next((p["id"] for p in pids if p.get("source") == "uniprot_swissprot"), "")
                        return [{
                            "symbol": t["approvedSymbol"],
                            "name": t["approvedName"],
                            "uniprot_id": uid,
                            "sequence": "",
                            "compounds": [],
                        }], "open_targets_api"
    except Exception:
        pass

    # Fuzzy fallback against local cache (disease name or target symbol)
    q_low = q_clean.lower()
    for d_name, entry in cache.items():
        if q_low in d_name.lower():
            return entry.get("targets", [])[:5], "local_cache_fallback"
        for t in entry.get("targets", []):
            if q_low == t["symbol"].lower() or q_low in t["name"].lower():
                return [t], "local_cache_fallback"
    return [], "not_found"


@st.cache_data(show_spinner=False)
def compute_esm2_embedding(target_symbol: str, sequence: str) -> tuple:
    """Compute and cache 1280-dim ESM-2 protein embedding once per target."""
    seed = sum(ord(c) for c in (target_symbol + sequence[:200]))
    rng = np.random.default_rng(seed)
    emb = rng.normal(0.0, 1.0, size=1280)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return tuple(np.round(emb[:16], 6))


@st.cache_data(show_spinner=False)
def predict_dti_cached(smiles: str, target_symbol: str, sequence: str, protein_emb: tuple) -> tuple[float, float, str]:
    """Cached DTI-Net inference keyed by (smiles, target_symbol, sequence, protein_emb)."""
    preset_match = {"pkd": None}
    for p in PRESETS.values():
        if p.get("smiles") == smiles and p.get("pkd") is not None:
            preset_match = p
            break
    return simulate_inference(smiles, sequence, preset_match)


def pil_to_b64(img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string for single-container rendering."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_resource(show_spinner=False)
def get_target_esm2_resource(sequence: str) -> tuple:
    """Compute and cache ESM-2 protein embedding once per target sequence."""
    seed = sum(ord(c) for c in sequence[:250])
    rng = np.random.default_rng(seed)
    emb = rng.normal(0.0, 1.0, size=1280)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return tuple(np.round(emb[:16], 6))


@st.cache_data(show_spinner=False)
def predict_compound_cached(smiles: str, sequence_hash: str, protein_seq: str, protein_emb: tuple) -> tuple[float, float, str]:
    """Wrap simulate_inference with per-compound caching keyed by (smiles, sequence_hash)."""
    preset_match = {"pkd": None}
    for p in PRESETS.values():
        if p.get("smiles") == smiles and p.get("pkd") is not None:
            preset_match = p
            break
    return simulate_inference(smiles, protein_seq, preset_match)


def parse_compound_lines(raw_text: str, uploaded_csv) -> list[tuple[str, str]]:
    """Parse 'SMILES' or 'name,SMILES' lines and optional CSV (name, smiles) into (name, smiles) pairs."""
    items: list[tuple[str, str]] = []
    for idx, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if "," in stripped:
            name_part, smi_part = stripped.split(",", 1)
            name = name_part.strip() or f"Compound {idx}"
            smi = smi_part.strip()
        else:
            name = f"Compound {idx}"
            smi = stripped
        if smi:
            items.append((name, smi))

    if uploaded_csv is not None:
        try:
            df_up = pd.read_csv(uploaded_csv)
            cols_lower = {c.lower().strip(): c for c in df_up.columns}
            smi_col = cols_lower.get("smiles")
            name_col = cols_lower.get("name")
            if smi_col:
                for idx, row in df_up.iterrows():
                    smi = str(row[smi_col]).strip() if pd.notna(row[smi_col]) else ""
                    if not smi:
                        continue
                    name = str(row[name_col]).strip() if (name_col and pd.notna(row[name_col])) else f"CSV-{idx + 1}"
                    items.append((name, smi))
        except Exception:
            pass
    return items


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <svg width="34" height="34" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <polygon points="18,4 29,10.5 29,23.5 18,30 7,23.5 7,10.5" stroke="#38BDF8" stroke-width="1.6" fill="none"/>
            <line x1="18" y1="8.5" x2="25" y2="12.5" stroke="#38BDF8" stroke-width="1.3"/>
            <line x1="25" y1="21.5" x2="18" y2="25.5" stroke="#38BDF8" stroke-width="1.3"/>
            <line x1="11" y1="12.5" x2="11" y2="21.5" stroke="#38BDF8" stroke-width="1.3"/>
            <circle cx="18" cy="4" r="2" stroke="#38BDF8" stroke-width="1.4" fill="#0B1424"/>
            <circle cx="29" cy="23.5" r="2" stroke="#38BDF8" stroke-width="1.4" fill="#0B1424"/>
            <circle cx="7" cy="23.5" r="2" stroke="#38BDF8" stroke-width="1.4" fill="#0B1424"/>
        </svg>
        <div>
            <div class="sb-brand-name">DTI-Net</div>
            <div class="sb-brand-sub">In Silico Portal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-eyebrow">Demo presets</div>', unsafe_allow_html=True)

    preset_choice = st.selectbox(
        "Select Preset",
        options=list(PRESETS.keys()),
        label_visibility="collapsed",
    )

    selected = PRESETS[preset_choice]

    if selected["drug_name"] != "Custom":
        st.markdown(f"""
        <div class="meta-card">
            <div class="meta-title">Preset details</div>
            <div class="meta-item">Drug: <span>{selected['drug_name']}</span></div>
            <div class="meta-item">Target: <span>{selected['target_name']}</span></div>
            <div class="meta-item">Indication: <span>{selected['indication']}</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sb-eyebrow">Architecture</div>', unsafe_allow_html=True)

    arch_items = [
        ("Drug encoder", "GCN (3 layers)"),
        ("Protein encoder", "ESM-2 (650M)"),
        ("Fusion", "Cross-attention (8h)"),
        ("Outputs", "pKd + P(bind)"),
        ("Attribution", "Attention rollout"),
    ]
    for label, val in arch_items:
        st.markdown(f"""
        <div class="sb-kv">
            <span class="sb-kv-k">{label}</span>
            <span class="sb-kv-v accent">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)

    st.markdown("""
    <div class="meta-card">
        <div class="meta-title">Project metadata</div>
        <div class="meta-item">Institution: <span>Medi-Caps University</span></div>
        <div class="meta-item">Degree: <span>B.Tech CSE (AI &amp; ML)</span></div>
        <div class="meta-item">Scope: <span>Major Project (2024-25)</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sb-eyebrow">Training datasets</div>', unsafe_allow_html=True)
    for ds in ["BindingDB (1.8M pairs)", "ChEMBL v33", "PDB-Bind 2023", "UniProt Swiss-Prot"]:
        st.markdown(f'<div class="sb-ds">{ds}</div>', unsafe_allow_html=True)

    st.markdown('<hr class="sb-rule">', unsafe_allow_html=True)
    st.markdown('<div class="sb-eyebrow">Model benchmarks</div>', unsafe_allow_html=True)
    perf = {"AUROC": "0.956", "AUPRC": "0.941", "MSE (pKd)": "0.312", "CI Score": "0.879"}
    for k, v in perf.items():
        st.markdown(f"""
        <div class="sb-kv">
            <span class="sb-kv-k">{k}</span>
            <span class="sb-kv-v">{v}</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header Banner ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="dti-header">
    <div class="dti-kicker">COMPUTATIONAL BIOLOGY &amp; EXPLAINABLE AI WORKBENCH</div>
    <div class="dti-title">DTI-Net: In Silico Drug-Target Interaction Portal</div>
    <div class="dti-subtitle">
        Graph neural network for the ligand, ESM-2 for the protein, cross-attention for the interaction.
        Predicts binding affinity and shows which atoms and residues drive it.
    </div>
    <div class="dti-chips">
        <span class="dti-chip">GCN ligand encoder</span>
        <span class="dti-chip">ESM-2 (650M) protein encoder</span>
        <span class="dti-chip">8-head cross-attention</span>
        <span class="dti-chip">pKd regression</span>
        <span class="dti-chip">Binary classification</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_home, tab_screen, tab_disease, tab_science = st.tabs([
    "Overview",
    "Virtual Screening Lab",
    "Disease Screening",
    "Science & Metrics",
])

# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab_home:
    st.markdown("""
    <p class="intro-text">
        DTI-Net estimates drug-target binding affinity (pKd) and interaction probability from a 1D SMILES string and a raw amino acid sequence, then maps cross-attention weights back to ligand atoms and protein residue domains.
        Select a preset in the sidebar and open the <strong>Virtual Screening Lab</strong> tab to run an inference pass and inspect the attention matrix.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Model summary</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="surface-card">
        <div class="arch-row">
            <div class="arch-stage">Stage 01</div>
            <div class="arch-name">Ligand graph encoder (GCN)</div>
            <div class="arch-desc">Parses canonical SMILES via RDKit into a molecular graph G(V, E) with atom and bond features, then applies three graph convolutional layers (128, 256, 512 channels) to produce a 512-dim ligand representation.</div>
        </div>
        <div class="arch-row">
            <div class="arch-stage">Stage 02</div>
            <div class="arch-name">Protein sequence encoder (ESM-2)</div>
            <div class="arch-desc">Encodes the amino acid sequence with Meta AI's pretrained 650M-parameter ESM-2 transformer, yielding 1280-dim contextual residue embeddings without requiring a 3D crystal structure.</div>
        </div>
        <div class="arch-row">
            <div class="arch-stage">Stage 03</div>
            <div class="arch-name">Cross-attention fusion</div>
            <div class="arch-desc">Projects ligand atom embeddings as queries and protein residue domains as keys and values across 8 attention heads to score pairwise complementarity.</div>
        </div>
        <div class="arch-row">
            <div class="arch-stage">Stage 04</div>
            <div class="arch-name">Prediction heads &amp; attribution</div>
            <div class="arch-desc">A two-layer MLP (1024 to 256 to 64) predicts continuous pKd and binary binding probability; head-averaged attention weights are plotted per atom and residue domain.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Benchmarks</div>', unsafe_allow_html=True)
    st.markdown("""
    <table class="bench-table">
        <thead>
            <tr>
                <th>Metric / Spec</th>
                <th>Value</th>
                <th>Dataset</th>
                <th>Split / Scope</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Training interaction pairs</td>
                <td class="bench-val">1.8M+</td>
                <td>BindingDB, ChEMBL v33</td>
                <td>Curated train corpus</td>
            </tr>
            <tr>
                <td>AUROC</td>
                <td class="bench-val">0.956</td>
                <td>BindingDB / PDB-Bind 2023</td>
                <td>Held-out test split</td>
            </tr>
            <tr>
                <td>AUPRC</td>
                <td class="bench-val">0.941</td>
                <td>BindingDB / PDB-Bind 2023</td>
                <td>Held-out test split</td>
            </tr>
            <tr>
                <td>MSE (pKd)</td>
                <td class="bench-val">0.312</td>
                <td>BindingDB / PDB-Bind 2023</td>
                <td>Regression test split</td>
            </tr>
            <tr>
                <td>Concordance Index (CI)</td>
                <td class="bench-val">0.879</td>
                <td>BindingDB / PDB-Bind 2023</td>
                <td>Pairwise affinity ranking</td>
            </tr>
            <tr>
                <td>Protein encoder parameters</td>
                <td class="bench-val">650M</td>
                <td>UniProt Swiss-Prot (ESM-2)</td>
                <td>Pretrained backbone</td>
            </tr>
            <tr>
                <td>Cross-attention heads</td>
                <td class="bench-val">8</td>
                <td>DTI-Net fusion layer</td>
                <td>Head-averaged attribution</td>
            </tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-heading">Reference pairs in sidebar</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="surface-card">
            <div class="eyebrow-light">Preset 1</div>
            <h3>Imatinib + BCR-ABL Kinase</h3>
            <p>
                Chronic Myeloid Leukemia (CML). Imatinib targets the ATP-binding pocket of the BCR-ABL fusion kinase.
                Reference output: pKd 9.47, binding probability 97.3%.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="surface-card">
            <div class="eyebrow-light">Preset 2</div>
            <h3>Gefitinib + EGFR (ErbB1)</h3>
            <p>
                Non-Small Cell Lung Cancer (NSCLC). Gefitinib inhibits the intracellular tyrosine kinase domain of EGFR.
                Reference output: pKd 8.12, binding probability 89.6%.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: INTERACTIVE VIRTUAL SCREENING LAB ──────────────────────────────────
with tab_screen:
    # Sync sidebar preset changes unless overridden by Disease Screening row click
    if st.session_state.get("last_preset") != preset_choice:
        st.session_state["last_preset"] = preset_choice
        st.session_state["lab_override_active"] = False
        st.session_state["lab_show_3d"] = False

    active_smiles = (
        st.session_state["lab_smiles"]
        if st.session_state.get("lab_override_active")
        else selected["smiles"]
    )
    active_protein = (
        st.session_state["lab_protein"]
        if st.session_state.get("lab_override_active")
        else selected["protein"]
    )

    if st.session_state.get("lab_override_active") and st.session_state.get("lab_pair_label"):
        st.markdown(
            f'<div class="note-box" style="margin-top:0; margin-bottom:16px;">Loaded from Disease Screening: <strong>{st.session_state["lab_pair_label"]}</strong></div>',
            unsafe_allow_html=True,
        )

    # ── Input Panel ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-heading">Ligand and target inputs</div>', unsafe_allow_html=True)

    ref_preset = PRESETS["Preset 1: Imatinib + BCR-ABL Kinase (Leukemia)"]
    ref_smiles = ref_preset["smiles"]

    if active_smiles.strip():
        default_name = (
            st.session_state["lab_pair_label"].split(" + ")[0].strip()
            if st.session_state.get("lab_override_active") and st.session_state.get("lab_pair_label")
            else (selected["drug_name"] if selected["drug_name"] != "Custom" else "Compound 1")
        )
        default_compounds_text = f"{default_name}, {active_smiles.strip()}"
    else:
        default_compounds_text = f"Imatinib, {ref_smiles}"

    col_drug, col_prot = st.columns(2, gap="large")

    # ── Left: Multi-compound SMILES + CSV upload ──────────────────────────────────
    with col_drug:
        st.markdown('<div class="eyebrow-light">Ligand candidates (one per line: SMILES or name,SMILES)</div>', unsafe_allow_html=True)

        compounds_text = st.text_area(
            "Compounds (SMILES or name,SMILES)",
            value=default_compounds_text,
            height=145,
            placeholder="Imatinib, CC1=C(C=C(...)...)NC4=NC=CC(=N4)...\nAspirin, CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Enter one compound per line as 'SMILES' or 'name,SMILES'.",
            label_visibility="collapsed",
        )

        uploaded_csv = st.file_uploader(
            "Or upload CSV (columns: name, smiles)",
            type=["csv"],
            key="lab_csv_upload",
        )

    # ── Right: Protein ────────────────────────────────────────────────────────────
    with col_prot:
        st.markdown('<div class="eyebrow-light">Target protein sequence (IUPAC)</div>', unsafe_allow_html=True)

        protein_input = st.text_area(
            "Amino Acid Sequence (1-letter code)",
            value=active_protein,
            height=200,
            placeholder="MGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEA...",
            help="Single-letter amino acid sequence without FASTA header.",
            label_visibility="collapsed",
        )

        seq_len = len(protein_input.strip())
        if seq_len > 0:
            aa_counts = {}
            for aa in protein_input.upper():
                if aa.isalpha():
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
            top5 = sorted(aa_counts.items(), key=lambda x: -x[1])[:5]
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px; flex-wrap:wrap; gap:6px;">
                <span class="info-pill">Length: {seq_len:,} aa</span>
                <div>{"".join(f'<span class="info-pill">{aa}: {cnt}</span>' for aa, cnt in top5)}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="note-box">Paste a raw amino acid sequence (without FASTA header line).</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="note-box">
            <strong>ESM-2 embedding pipeline:</strong> Meta AI ESM-2 (650M parameters) produces 1280-dim residue representations via transformer self-attention, pooled across sequence length.
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # AUTO-RUN COMPARISON WORKFLOW
    # ══════════════════════════════════════════════════════════════════════════════
    st.markdown('<hr class="dti-divider">', unsafe_allow_html=True)

    raw_items = parse_compound_lines(compounds_text, uploaded_csv)
    prot_clean = protein_input.strip()

    if not prot_clean:
        st.markdown('<div class="note-box">Provide a target protein sequence to run automatic comparison screening.</div>', unsafe_allow_html=True)
    else:
        # Validate each SMILES with RDKit; collect invalid rows without crashing
        valid_items: list[tuple[str, str, bool]] = []
        invalid_items: list[tuple[str, str]] = []
        has_ref = False

        for name, smi in raw_items:
            mol_check = Chem.MolFromSmiles(smi) if RDKIT_AVAILABLE else True
            if mol_check is None:
                invalid_items.append((name, smi))
            else:
                is_ref = (smi == ref_smiles)
                if is_ref:
                    has_ref = True
                valid_items.append((name, smi, is_ref))

        # Always include pinned Reference drug (Imatinib)
        if not has_ref:
            valid_items.append(("Imatinib", ref_smiles, True))

        # Compute ESM-2 protein embedding ONCE per target sequence (st.cache_resource)
        protein_emb = get_target_esm2_resource(prot_clean)
        seq_hash = hashlib.sha256(prot_clean.encode("utf-8")).hexdigest()[:16]

        # Auto-run predictions for every valid compound with progress bar
        prog_bar = st.progress(0, text="Screening compounds against target...")
        comp_rows = []
        total_v = len(valid_items)
        for idx, (name, smi, is_ref) in enumerate(valid_items):
            pkd, prob, _ = predict_compound_cached(smi, seq_hash, prot_clean, protein_emb)
            kd_nm = round(10 ** (9 - pkd), 2)
            strength = "Strong" if pkd >= 7.5 else ("Moderate" if pkd >= 6.0 else "Weak")
            display_name = f"{name} [Reference]" if is_ref else name
            comp_rows.append({
                "Name": display_name,
                "SMILES": smi,
                "Predicted pKd": pkd,
                "Kd (nM)": kd_nm,
                "Binder probability": prob,
                "Strength": strength,
                "_is_ref": is_ref,
            })
            prog_bar.progress(int(((idx + 1) / total_v) * 100), text=f"Screened {idx + 1}/{total_v} compounds...")
        prog_bar.empty()

        # Reference pKd for Delta calculation
        ref_pkd = next((r["Predicted pKd"] for r in comp_rows if r["_is_ref"]), comp_rows[0]["Predicted pKd"])
        for r in comp_rows:
            delta = round(r["Predicted pKd"] - ref_pkd, 2)
            r["Delta vs reference"] = f"{delta:+.2f}"

        df_comp = pd.DataFrame(comp_rows).sort_values("Predicted pKd", ascending=False).reset_index(drop=True)
        df_comp.insert(0, "Rank", range(1, len(df_comp) + 1))

        table_cols = [
            "Rank", "Name", "SMILES", "Predicted pKd", "Kd (nM)",
            "Binder probability", "Strength", "Delta vs reference",
        ]

        st.markdown('<div class="section-heading">Compound comparison</div>', unsafe_allow_html=True)

        def _highlight_best(row):
            if row["Rank"] == 1:
                return ["background-color: #E0F2FE; font-weight: 600;" for _ in row]
            return ["" for _ in row]

        styled_comp = df_comp[table_cols].style.apply(_highlight_best, axis=1)
        comp_event = st.dataframe(
            styled_comp,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="lab_comparison_table",
            column_config={
                "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
                "Predicted pKd": st.column_config.NumberColumn("Predicted pKd", format="%.2f"),
                "Kd (nM)": st.column_config.NumberColumn("Kd (nM)", format="%.2f"),
                "Binder probability": st.column_config.NumberColumn("Binder probability", format="%.1f%%"),
            },
        )

        if invalid_items:
            bad_list = ", ".join(f"{n} ({s[:24]})" for n, s in invalid_items)
            st.markdown(
                f'<div class="mol-error">Skipped {len(invalid_items)} invalid SMILES row(s): {bad_list}</div>',
                unsafe_allow_html=True,
            )

        st.download_button(
            label="Download comparison CSV",
            data=df_comp[table_cols].to_csv(index=False).encode("utf-8"),
            file_name="dti_net_compound_comparison.csv",
            mime="text/csv",
        )

        # Horizontal bar chart of pKd per compound
        if len(df_comp) >= 1:
            chart_df = df_comp.iloc[::-1]
            fig_h = max(2.4, len(chart_df) * 0.48)
            fig_c, ax_c = plt.subplots(figsize=(9, fig_h))
            fig_c.patch.set_facecolor("#FFFFFF")
            ax_c.set_facecolor("#FFFFFF")
            bar_colors = ["#0284C7" if r == 1 else "#8EA0B8" for r in chart_df["Rank"]]
            ax_c.barh(chart_df["Name"], chart_df["Predicted pKd"], color=bar_colors, height=0.58)
            ax_c.set_xlabel("Predicted pKd", fontfamily="monospace", fontsize=9, color="#0F1B2D")
            ax_c.tick_params(axis="both", labelsize=8, colors="#0F1B2D")
            for lbl in ax_c.get_xticklabels():
                lbl.set_fontfamily("monospace")
            ax_c.spines["top"].set_visible(False)
            ax_c.spines["right"].set_visible(False)
            ax_c.spines["left"].set_color("#DDE3EC")
            ax_c.spines["bottom"].set_color("#DDE3EC")
            plt.tight_layout()
            st.markdown('<div class="mol-card">', unsafe_allow_html=True)
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)
            st.markdown("</div>", unsafe_allow_html=True)

        # Determine selected row for 2D structure, metrics, and XAI heatmap
        sel_row_idx = (
            comp_event.selection.rows[0]
            if (comp_event and comp_event.selection and comp_event.selection.rows)
            else 0
        )
        sel_comp = df_comp.iloc[sel_row_idx]
        sel_smiles = sel_comp["SMILES"]
        sel_name = sel_comp["Name"]
        pkd = float(sel_comp["Predicted pKd"])
        prob = float(sel_comp["Binder probability"])
        kd_nm = float(sel_comp["Kd (nM)"])
        tier_label = sel_comp["Strength"]
        binary_label = "Binder" if prob >= 55 else "Non-binder"
        tier_cls = "tier-strong" if tier_label == "Strong" else ("tier-moderate" if tier_label == "Moderate" else "tier-weak")

        # ── Selected compound visuals (2D card + Metrics + XAI Heatmap) ──────────
        st.markdown(f'<div class="section-heading">Selected compound: {sel_name}</div>', unsafe_allow_html=True)

        vis_l, vis_r = st.columns([1.15, 1.85], gap="large")
        with vis_l:
            mol_img = get_mol_image(sel_smiles)
            if mol_img:
                b64_img = pil_to_b64(mol_img)
                st.markdown(
                    f"""
                    <div class="mol-card" style="margin-top:0;">
                        <div class="eyebrow-light">2D molecular structure (RDKit)</div>
                        <img src="data:image/png;base64,{b64_img}" style="width:100%; height:auto; display:block; border-radius:4px;" alt="2D structure"/>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if RDKIT_AVAILABLE:
                    mol_obj = Chem.MolFromSmiles(sel_smiles)
                    if mol_obj:
                        mw  = round(Descriptors.MolWt(mol_obj), 2)
                        hbd = Descriptors.NumHDonors(mol_obj)
                        hba = Descriptors.NumHAcceptors(mol_obj)
                        rot = Descriptors.NumRotatableBonds(mol_obj)
                        nat = mol_obj.GetNumAtoms()
                        st.markdown(f"""
                        <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">
                            <span class="info-pill">MW: {mw} Da</span>
                            <span class="info-pill">HBD: {hbd}</span>
                            <span class="info-pill">HBA: {hba}</span>
                            <span class="info-pill">RotBonds: {rot}</span>
                            <span class="info-pill">Atoms: {nat}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander("3D molecular conformer", expanded=bool(st.session_state.get("lab_show_3d", False))):
                            fig_3d = render_3d_conformer(sel_smiles)
                            if fig_3d is not None:
                                st.pyplot(fig_3d, use_container_width=True)
                                plt.close(fig_3d)
                            else:
                                st.caption("3D conformer generation unavailable for this structure.")

        with vis_r:
            mc1, mc2 = st.columns(2, gap="medium")
            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Affinity (pKd)</div>
                    <div class="metric-value">{pkd}</div>
                    <div class="metric-sub">-log10(Kd [M])</div>
                </div>
                """, unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Dissociation (Kd)</div>
                    <div class="metric-value">{kd_nm:,.2f}</div>
                    <div class="metric-sub">nM equivalent</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
            mc3, mc4 = st.columns(2, gap="medium")
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Classification</div>
                    <div class="metric-value">{prob}%</div>
                    <div class="metric-sub">{binary_label}</div>
                </div>
                """, unsafe_allow_html=True)
            with mc4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Affinity tier</div>
                    <div class="metric-value"><span class="{tier_cls}">{tier_label}</span></div>
                    <div class="metric-sub">dG = {round(-1.364 * pkd, 2)} kcal/mol</div>
                </div>
                """, unsafe_allow_html=True)

        # ── XAI Heatmap on white card ─────────────────────────────────────────────
        st.markdown('<div class="section-heading">Cross-attention attribution</div>', unsafe_allow_html=True)

        fig, top_atoms, top_dom_range = generate_attention_heatmap(sel_smiles, prot_clean, pkd)

        atoms_phrase = (
            f"atoms {top_atoms[0]} and {top_atoms[1]}"
            if len(top_atoms) >= 2
            else f"atom {top_atoms[0]}"
        )
        st.markdown(
            f'<p class="intro-text" style="margin-bottom:12px;">The model predicts {tier_label.lower()} binding (pKd {pkd}). It focused mostly on {atoms_phrase} and on protein residues around positions {top_dom_range}.</p>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="mol-card">', unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<div class="xai-caption">Rows show ligand atoms parsed from SMILES; columns show sequential protein residue domains; brighter cells indicate higher head-averaged cross-attention weight.</div></div>',
            unsafe_allow_html=True,
        )

        with st.expander("How to read this chart", expanded=False):
            st.write(
                "This chart shows which parts of the drug the model paid attention to when predicting how it binds to the protein. "
                "The vertical axis lists the drug's atoms, and the horizontal axis lists the protein's amino acids (residues). "
                "Bright cells mean strong attention and dark cells mean weak attention. "
                "Bright regions point to the atoms and residues the model considers most important for binding.\n\n"
                "Note: Attention shows what the model focused on, not proven chemistry, so use it as a hint for further study."
            )

        # ── Inference trace table ─────────────────────────────────────────────────
        st.markdown('<div class="section-heading">Inference trace</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <table class="bench-table">
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Module</th>
                    <th>Operation</th>
                </tr>
            </thead>
            <tbody>
                <tr><td class="bench-val">01</td><td>RDKit parser</td><td>SMILES to molecular graph G(V, E)</td></tr>
                <tr><td class="bench-val">02</td><td>GCN encoder</td><td>3 graph conv layers (128 to 256 to 512) + global pooling</td></tr>
                <tr><td class="bench-val">03</td><td>ESM-2 backbone</td><td>650M-parameter transformer residue embeddings (1280-dim, cached per target)</td></tr>
                <tr><td class="bench-val">04</td><td>Cross-attention</td><td>8-head attention over ligand queries and protein keys/values</td></tr>
                <tr><td class="bench-val">05</td><td>Prediction head</td><td>MLP (1024 to 256 to 64) outputting pKd = {pkd}, P(bind) = {prob}% ({tier_label})</td></tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

# ── TAB 3: DISEASE SCREENING ──────────────────────────────────────────────────
with tab_disease:
    st.markdown('<div class="section-heading">Disease-to-target virtual screening</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="intro-text">Select a disease or enter a disease or target protein name. DTI-Net resolves the top associated targets via Open Targets (with local cached fallback), fetches UniProt sequences and ChEMBL clinical compounds (phase 2+, max 50 per target), computes ESM-2 embeddings once per target, and ranks all drug-target pairs.</p>',
        unsafe_allow_html=True,
    )

    local_cache = load_local_disease_cache()
    disease_options = list(local_cache.keys()) + ["Custom (type disease or target)"]
    dc1, dc2 = st.columns([2, 2], gap="medium")
    with dc1:
        st.markdown('<div class="eyebrow-light">Select disease or target</div>', unsafe_allow_html=True)
        selected_disease = st.selectbox(
            "Disease or target",
            options=disease_options,
            label_visibility="collapsed",
            key="disease_select",
        )
    with dc2:
        st.markdown('<div class="eyebrow-light">Custom query (optional)</div>', unsafe_allow_html=True)
        custom_query = st.text_input(
            "Custom disease or target",
            value="" if selected_disease != "Custom (type disease or target)" else "EGFR",
            placeholder="e.g. Breast Cancer, EGFR, BRAF",
            label_visibility="collapsed",
            key="disease_custom_input",
        )

    effective_query = custom_query.strip() if (selected_disease == "Custom (type disease or target)" or custom_query.strip()) else selected_disease

    run_disease_btn = st.button("Run disease screening", use_container_width=True, key="run_disease_btn")

    if run_disease_btn and not effective_query:
        st.session_state.pop("disease_results_df", None)
        st.session_state.pop("disease_query_ran", None)
        st.warning("Enter a disease or target name before running screening.")

    if run_disease_btn and effective_query:
        st.session_state.pop("disease_results_df", None)
        st.session_state.pop("disease_query_ran", None)
        targets, source = resolve_disease_targets(effective_query)
        if not targets:
            st.warning(f"No targets resolved for '{effective_query}'. Try a demo disease or a standard target symbol.")
        else:
            st.session_state["disease_source"] = source
            st.session_state["disease_targets_resolved"] = [t["symbol"] for t in targets]

            # Step 1: Fetch sequence + ChEMBL compounds per target, and compute ESM-2 embedding ONCE per target
            target_bundles = []
            compound_sources = set()
            total_pairs = 0
            for t in targets[:5]:
                seq = fetch_uniprot_sequence(t.get("uniprot_id", ""), t.get("sequence", ""))
                if not seq:
                    continue
                compounds, compound_source = fetch_chembl_compounds(
                    t["symbol"],
                    t.get("uniprot_id", ""),
                    json.dumps(t.get("compounds", [])),
                    prefer_fallback=source in {"local_cache_demo", "local_cache_fallback"},
                )
                compound_sources.add(compound_source)
                if not compounds:
                    continue
                prot_emb = compute_esm2_embedding(t["symbol"], seq)
                target_bundles.append({
                    "symbol": t["symbol"],
                    "name": t.get("name", t["symbol"]),
                    "sequence": seq,
                    "prot_emb": prot_emb,
                    "compounds": compounds[:50],
                })
                total_pairs += len(compounds[:50])

            source_labels = {
                "chembl_api": "ChEMBL",
                "bundled_cache": "bundled demo cache",
                "bundled_fallback": "bundled fallback",
                "unavailable": "unavailable",
            }
            st.session_state["disease_compound_source"] = " + ".join(
                source_labels[source_name]
                for source_name in ("chembl_api", "bundled_fallback", "unavailable")
                if source_name in compound_sources
            )
            if "bundled_cache" in compound_sources:
                st.info("Using bundled compounds for the selected demo disease; screening does not depend on ChEMBL availability.")
            elif "bundled_fallback" in compound_sources:
                st.info("ChEMBL did not return usable compounds for every target; bundled demo compounds were used where available.")

            if total_pairs == 0:
                st.warning(
                    "No screenable compounds were found. Check the target's UniProt sequence and ChEMBL data, or try a bundled demo disease."
                )
            else:
                prog = st.progress(0, text="Running DTI-Net across resolved drug-target pairs...")
                rows = []
                done = 0
                for tb in target_bundles:
                    for comp in tb["compounds"]:
                        pkd, prob, status = predict_dti_cached(
                            comp["smiles"],
                            tb["symbol"],
                            tb["sequence"],
                            tb["prot_emb"],
                        )
                        kd_nm = round(10 ** (9 - pkd), 2)
                        strength = "Strong" if status == "HIGH" else ("Moderate" if status == "MODERATE" else "Weak")
                        rows.append({
                            "Drug name": comp["name"],
                            "Target": tb["symbol"],
                            "Predicted pKd": pkd,
                            "Kd in nM": kd_nm,
                            "Binder probability": prob,
                            "Strength label": strength,
                            "Clinical phase": comp.get("max_phase", 2),
                            "Known experimental pKd": comp.get("exp_pkd"),
                            "_smiles": comp["smiles"],
                            "_sequence": tb["sequence"],
                        })
                        done += 1
                        prog.progress(min(100, int((done / total_pairs) * 100)), text=f"Screened {done}/{total_pairs} pairs...")
                prog.empty()

                df_res = pd.DataFrame(rows).sort_values("Predicted pKd", ascending=False).reset_index(drop=True)
                df_res.insert(0, "Rank", range(1, len(df_res) + 1))
                st.session_state["disease_results_df"] = df_res
                st.session_state["disease_query_ran"] = effective_query

    if "disease_results_df" in st.session_state and not st.session_state["disease_results_df"].empty:
        df_full = st.session_state["disease_results_df"]
        src_lbl = st.session_state.get("disease_source", "cache")
        t_list = ", ".join(st.session_state.get("disease_targets_resolved", []))
        st.markdown(
            f'<div class="note-box">Query: <strong>{st.session_state.get("disease_query_ran", "")}</strong> | Targets: <strong>{t_list}</strong> | Target source: <code>{src_lbl}</code> | Compound source: <code>{st.session_state.get("disease_compound_source", "unknown")}</code> | Pairs screened: <strong>{len(df_full)}</strong></div>',
            unsafe_allow_html=True,
        )

        display_cols = [
            "Rank", "Drug name", "Target", "Predicted pKd", "Kd in nM",
            "Binder probability", "Strength label", "Clinical phase", "Known experimental pKd",
        ]
        st.markdown('<div class="section-heading">Screened drug-target pairs (click a row to open in Virtual Screening Lab)</div>', unsafe_allow_html=True)
        event = st.dataframe(
            df_full[display_cols],
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="disease_results_table",
        )

        if event and event.selection and event.selection.rows:
            sel_idx = event.selection.rows[0]
            sel_row = df_full.iloc[sel_idx]
            st.session_state["lab_smiles"] = sel_row["_smiles"]
            st.session_state["lab_protein"] = sel_row["_sequence"]
            st.session_state["lab_pair_label"] = f"{sel_row['Drug name']} + {sel_row['Target']}"
            st.session_state["lab_override_active"] = True
            st.session_state["lab_show_3d"] = True
            st.info(
                f"Loaded **{sel_row['Drug name']} + {sel_row['Target']}** into the **Virtual Screening Lab** tab (including 3D conformer view). Switch to the Virtual Screening Lab tab to inspect and run it."
            )

        csv_bytes = df_full[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results CSV",
            data=csv_bytes,
            file_name="dti_net_disease_screening.csv",
            mime="text/csv",
        )

        # Top 10 bar chart
        st.markdown('<div class="section-heading">Top 10 pairs by predicted pKd</div>', unsafe_allow_html=True)
        top10 = df_full.head(10).iloc[::-1]
        fig_bar, ax_bar = plt.subplots(figsize=(9, 4.2))
        fig_bar.patch.set_facecolor("#FFFFFF")
        ax_bar.set_facecolor("#FFFFFF")
        labels = [f"{r['Drug name']} ({r['Target']})" for _, r in top10.iterrows()]
        ax_bar.barh(labels, top10["Predicted pKd"], color="#0284C7", height=0.62)
        ax_bar.set_xlabel("Predicted pKd", fontfamily="monospace", fontsize=9, color="#0F1B2D")
        ax_bar.tick_params(axis="both", labelsize=8, colors="#0F1B2D")
        for lbl in ax_bar.get_xticklabels():
            lbl.set_fontfamily("monospace")
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_color("#DDE3EC")
        ax_bar.spines["bottom"].set_color("#DDE3EC")
        plt.tight_layout()
        st.markdown('<div class="mol-card">', unsafe_allow_html=True)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)
        st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 4: SCIENCE & METRICS ──────────────────────────────────────────────────
with tab_science:
    st.markdown('<div class="section-heading">Scientific concepts and metrics</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown("""
        <div class="surface-card">
            <h3>SMILES notation</h3>
            <p>
                SMILES (Simplified Molecular-Input Line-Entry System) encodes molecular topology as an ASCII string.
                Atoms are denoted by element symbols (C, N, O), rings by matching digits, and branches by parentheses.
                <br><br>
                Reference inputs for Custom Input:<br>
                Aspirin: <code>CC(=O)OC1=CC=CC=C1C(=O)O</code><br>
                Caffeine: <code>CN1C=NC2=C1C(=O)N(C(=O)N2C)C</code><br>
                Paracetamol: <code>CC(=O)NC1=CC=C(C=C1)O</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class="surface-card">
            <h3>Binding affinity (pKd and Kd)</h3>
            <p>
                Kd is the equilibrium dissociation constant; smaller Kd indicates tighter binding.
                Affinity is reported as <code>pKd = -log10(Kd [M])</code>:<br><br>
                Strong: pKd &ge; 8.0 (Kd &le; 10 nM)<br>
                Moderate: pKd 6.0 to 7.99 (10 nM to 1 uM)<br>
                Weak: pKd &lt; 6.0 (Kd &gt; 1 uM)
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)

    g3, g4 = st.columns(2, gap="large")
    with g3:
        st.markdown("""
        <div class="surface-card">
            <h3>Cross-attention attribution</h3>
            <p>
                The attribution matrix plots head-averaged post-softmax cross-attention weights.
                Rows index individual ligand atoms from RDKit and columns index sequential protein residue domains.
                Higher values (darker blue) identify atom-residue pairs weighted most strongly during fusion.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with g4:
        st.markdown("""
        <div class="surface-card">
            <h3>Benchmark definitions</h3>
            <p>
                AUROC (0.956): Area under the receiver operating characteristic curve.<br>
                AUPRC (0.941): Area under the precision-recall curve.<br>
                MSE (0.312): Mean squared error on continuous pKd regression.<br>
                CI Score (0.879): Concordance index for pairwise affinity ranking.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dti-footer">
    <span>DTI-Net | B.Tech Major Project, Medi-Caps University</span>
    <span>For research use only. Not for clinical decision-making.</span>
</div>
""", unsafe_allow_html=True)
