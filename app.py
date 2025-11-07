import streamlit as st
from pathlib import Path
import importlib

st.set_page_config(page_title="Analisis Sentimen", page_icon="🤖")

ICON_DIR = Path(__file__).parent / "icons"

PAGES = [
    {"name": "beranda", "title": "Beranda", "file": "1_Beranda", "icon": ICON_DIR / "house-solid-full.svg"},
    {"name": "upload_data", "title": "Upload Data", "file": "2_Upload_Data", "icon": ICON_DIR / "upload-solid-full.svg"},
    {"name": "klasifikasi", "title": "Klasifikasi", "file": "3_Klasifikasi", "icon": ICON_DIR / "robot-solid-full.svg"},
    {"name": "evaluasi", "title": "Evaluasi", "file": "4_Evaluasi", "icon": ICON_DIR / "list-solid-full.svg"},
    {"name": "cek_sentimen", "title": "Cek Sentimen", "file": "5_Cek_Sentimen", "icon": ICON_DIR / "spell-check-solid-full.svg"},
]

if "page" not in st.session_state:
    st.session_state.page = PAGES[0]["name"]

# logo
logo_svg = ""
try:
    with open(ICON_DIR / "keyboard-solid-full.svg", "r", encoding="utf-8") as f:
        logo_svg = f.read()
except:
    logo_svg = ""

st.sidebar.markdown(
    f"""
    <div style="display:flex; align-items:center; margin-bottom: 10px;">
        <div style="width:32px; height:32px; margin-right:10px;">{logo_svg}</div>
        <div style="font-weight:bold; font-size:20px;">Analisis Sentimen</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Menu")

for p in PAGES:
    icon_svg = ""
    try:
        with open(p["icon"], "r", encoding="utf-8") as f:
            icon_svg = f.read()
    except:
        icon_svg = ""

    col1, col2 = st.sidebar.columns([1, 5])
    with col1:
        st.markdown(icon_svg, unsafe_allow_html=True)
    with col2:
        if st.button(p["title"], key=f"menu_{p['name']}", use_container_width=True):
            st.session_state.page = p["name"]

st.sidebar.markdown("---")
#st.sidebar.caption("📌 Analisis Sentimen - Versi 1.0")

selected = next((p for p in PAGES if p["name"] == st.session_state.page), PAGES[0])

page_module = importlib.import_module(selected["file"])

if hasattr(page_module, "run") and callable(page_module.run):
    page_module.run()
else:
    st.error(f"Modul {selected['file']} tidak punya fungsi run().")
