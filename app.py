"""
PaletteCraft — AI-Powered Paint Mixing Assistant for Artists
============================================================
Main Streamlit application.

Features:
  • Auto white-balance calibration (Gray-World + Max-RGB)
  • K-Means dominant colour palette extraction
  • Kubelka-Munk subtractive mixing recipe optimisation
  • Delta-E paint-map visual overlay
  • Exportable recipes and inventory management
  • Auto-recipe generation on colour selection
  • Native Streamlit buttons for smooth state transitions
"""

import io
import json
import uuid
import copy 

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError 

from utils.calibration import auto_white_balance, get_image_array
from utils.color_utils import hex_to_rgb, rgb_to_hex, rgb_to_lab, delta_e_76
from utils.kmeans_palette import extract_palette, get_pixel_color
from utils.mixing_engine import find_best_recipe
from utils.paint_map import generate_paint_map, get_match_percentage, NEON_COLOURS

# ══════════════════════════════ PAGE CONFIG ═══════════════════════════════════
st.set_page_config(
    page_title="PaletteCraft",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════ CUSTOM CSS ════════════════════════════════════
st.markdown("""
<style>
/* ── Header ── */
.pc-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4);
}
.pc-header h1 { margin: 0; font-size: 2rem; }
.pc-header p  { margin: 0; opacity: 0.85; font-size: 0.95rem; }

/* ── Inventory card ── */
.inv-card {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 12px;
    padding: 10px;
    text-align: center;
    margin-bottom: 6px;
    transition: box-shadow .2s;
}
.inv-card:hover { box-shadow: 0 3px 12px rgba(0,0,0,0.12); }

/* ── Recipe card ── */
.recipe-block {
    background: #f4f6ff;
    border-left: 5px solid #667eea;
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0;
    font-size: 1rem;
}

/* ── Info box ── */
.info-row {
    display: flex; align-items: center; gap: 10px;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 8px 0;
}
.dot-lg {
    width: 36px; height: 36px;
    border-radius: 50%;
    border: 2px solid #555;
    flex-shrink: 0;
}

/* ── Progress tweak ── */
div[data-testid="stMetricValue"] { font-size: 1.4rem !important; }

/* ── Mobile ── */
@media (max-width: 768px) {
    .pc-header h1 { font-size: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════ DEFAULT DATA ══════════════════════════════════
DEFAULT_PAINTS = [
    {"id": "p01", "name": "Lemon Yellow",      "hex": "#AEA53E", "rgb": [174, 165, 62]},
    {"id": "p02", "name": "Orange",            "hex": "#A75E2A", "rgb": [167, 94, 42]},
    {"id": "p03", "name": "Deep Brilliant Purple","hex": "#98517D", "rgb": [152, 81, 125]},
    {"id": "p04", "name": "Crimson",           "hex": "#843734", "rgb": [132, 55, 52]},
    {"id": "p05", "name": "Ultramarine Blue",  "hex": "#4B4978", "rgb": [75, 73, 120]},
    {"id": "p06", "name": "Burnt Sienna",      "hex": "#53413D", "rgb": [83, 65, 61]},
    {"id": "p07", "name": "Sap Green",         "hex": "#24261B", "rgb": [36, 38, 27]},
    {"id": "p08", "name": "Light Green",       "hex": "#223425", "rgb": [34, 52, 37]},
    {"id": "p09", "name": "White",             "hex": "#7F8283", "rgb": [127, 130, 131]},
    {"id": "p10", "name": "Black",             "hex": "#100F12", "rgb": [16, 15, 18]},
    {"id": "p11", "name": "Silver",            "hex": "#606365", "rgb": [96, 99, 101]},
    {"id": "p12", "name": "Gold",              "hex": "#5B513E", "rgb": [91, 81, 62]},
]

# ══════════════════════════════ SESSION STATE ═════════════════════════════════
def _init():
    defaults = {
        "inventory"         : copy.deepcopy(DEFAULT_PAINTS),  
        "original_image"    : None,
        "calibrated_array"  : None,
        "cal_method"        : "gray_world",
        "palette"           : [],
        "selected_color"    : None,
        "selected_color_name": "—",
        "recipe"            : None,
        "paint_map_image"   : None,
        "threshold"         : 8.0,
        "max_colors"        : 3,
        "last_image_fingerprint": None, # Used to prevent re-processing on clicks
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ══════════════════════════════ UTILITY FNS ══════════════════════════════════

def _pil_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _sel_hex() -> str:
    c = st.session_state.selected_color
    return rgb_to_hex(*c) if c else "#888888"


def _color_block(hex_c: str, w: int = 50, h: int = 50, radius: int = 10) -> str:
    return (
        f'<div style="width:{w}px;height:{h}px;background:{hex_c};'
        f'border-radius:{radius}px;border:2px solid #555;'
        f'display:inline-block;box-shadow:0 2px 6px rgba(0,0,0,.25);"></div>'
    )


def _set_selected(rgb: list, name: str = "Custom"):
    st.session_state.selected_color = rgb
    st.session_state.selected_color_name = name
    st.session_state.paint_map_image = None
    
    if st.session_state.inventory and rgb is not None:
        with st.spinner("🧪 Auto-calculating recipe..."):
            recipe = find_best_recipe(
                rgb,
                st.session_state.inventory,
                max_colors=st.session_state.max_colors,
            )
            st.session_state.recipe = recipe


def _process_image(source):
    try:
        img = Image.open(source).convert("RGB")
    except Exception as e:
        st.error("⚠️ The uploaded file is corrupted or not a valid image format.")
        return

    arr = get_image_array(img, max_size=1200)
    st.session_state.original_image = Image.fromarray(arr)

    with st.spinner("Calibrating colours…"):
        cal = auto_white_balance(arr, method=st.session_state.cal_method)
    st.session_state.calibrated_array = cal

    with st.spinner("Extracting palette…"):
        palette = extract_palette(cal, n_colors=5)
    st.session_state.palette = palette

    if palette:
        _set_selected(palette[0], "Dominant Colour #1")

    st.session_state.paint_map_image = None


# ══════════════════════════════ SIDEBAR ══════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎨 PaletteCraft")
    n_inv = len(st.session_state.inventory)
    st.caption(f"🖌️ {n_inv} paint{'s' if n_inv != 1 else ''} in inventory")
    st.divider()

    with st.expander("➕ Add a Paint"):
        new_name = st.text_input("Paint name", placeholder="e.g. Naples Yellow")
        new_hex  = st.color_picker("Colour", "#FF6B00")
        if st.button("Add to Inventory", width="stretch"):
            if new_name.strip():
                rgb_val = list(hex_to_rgb(new_hex))
                st.session_state.inventory.append({
                    "id" : str(uuid.uuid4())[:8],
                    "name": new_name.strip(),
                    "hex" : new_hex,
                    "rgb" : rgb_val,
                })
                st.success(f"Added {new_name.strip()}")
                st.rerun()
            else:
                st.warning("Please enter a paint name.")

    with st.expander("🗂️ Add Preset Paint"):
        existing_names = {p["name"] for p in st.session_state.inventory}
        available = [p for p in DEFAULT_PAINTS if p["name"] not in existing_names]
        if available:
            preset = st.selectbox(
                "Choose preset:", [p["name"] for p in available], key="preset_sel"
            )
            if st.button("Add Preset", width="stretch"):
                for p in available:
                    if p["name"] == preset:
                        st.session_state.inventory.append(copy.deepcopy(p))
                        st.rerun()
        else:
            st.info("All presets already in inventory!")

    st.divider()

    st.markdown("### My Paint Box")
    for paint in list(st.session_state.inventory):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin:2px 0;">'
                f'<div style="width:18px;height:18px;background:{paint["hex"]};'
                f'border-radius:50%;border:1px solid #aaa;flex-shrink:0;"></div>'
                f'<span style="font-size:13px;">{paint["name"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("✕", key=f"del_{paint['id']}", help=f"Remove {paint['name']}"):
                st.session_state.inventory = [
                    p for p in st.session_state.inventory if p["id"] != paint["id"]
                ]
                st.rerun()

    st.divider()

    with st.expander("⚙️ Calibration Settings"):
        method = st.radio(
            "Auto White-Balance:",
            ["gray_world", "max_rgb", "combined"],
            format_func=lambda x: {
                "gray_world": "Gray-World (default)",
                "max_rgb"   : "Max-RGB / White Patch",
                "combined"  : "Hybrid (70/30 blend)",
            }[x],
            index=["gray_world", "max_rgb", "combined"].index(st.session_state.cal_method),
        )
        if method != st.session_state.cal_method:
            st.session_state.cal_method = method
            if st.session_state.calibrated_array is not None:
                orig = np.array(st.session_state.original_image)
                cal  = auto_white_balance(orig, method=method)
                st.session_state.calibrated_array = cal
                st.session_state.palette = extract_palette(cal, n_colors=5)
                st.session_state.paint_map_image = None
                st.rerun()


# ══════════════════════════════ MAIN HEADER ═══════════════════════════════════
st.markdown("""
<div class="pc-header">
  <h1>🎨 PaletteCraft</h1>
  <p>Paint mixing assistant — from photo to physical pigment recipe</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════ TABS ══════════════════════════════════════════
tab_img, tab_recipe, tab_map, tab_inv = st.tabs([
    "🖼️  Image & Palette",
    "🧪  Recipe Builder",
    "🗺️  Paint Map",
    "🎨  My Inventory",
])


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — IMAGE & PALETTE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_img:
    st.markdown("### 📸 Upload Your Inspiration Photo")

    input_method = st.radio(
        "Choose image source:", 
        ["Upload a file", "Use camera"], 
        horizontal=True,
        label_visibility="collapsed"
    )

    source = None
    if input_method == "Upload a file":
        source = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
    else:
        source = st.camera_input("📷 Take a photo", label_visibility="visible")

    if source:
        # Create a unique fingerprint so we don't re-process on every button click
        file_fingerprint = f"{source.name}_{source.size}"
        if st.session_state.get("last_image_fingerprint") != file_fingerprint:
            _process_image(source)
            st.session_state["last_image_fingerprint"] = file_fingerprint

    if st.session_state.calibrated_array is not None:
        arr = st.session_state.calibrated_array
        cal_img = Image.fromarray(arr)

        st.markdown("---")
        col_orig, col_cal = st.columns(2)
        with col_orig:
            st.markdown("**Original Photo**")
            st.image(st.session_state.original_image, width="stretch")
        with col_cal:
            st.markdown("**✨ Calibrated Photo**")
            st.image(cal_img, width="stretch")

        st.markdown("---")
        st.markdown("### 🎨 Dominant Colour Palette")
        st.caption("👆 **Click any colour swatch below** to select it. The recipe will auto-calculate!")

        if st.session_state.palette:
            pal_cols = st.columns(len(st.session_state.palette))
            for i, rgb in enumerate(st.session_state.palette):
                hx = rgb_to_hex(*rgb)
                # Safely compare hex codes to ensure UI correctly highlights the choice
                is_sel = (hx == _sel_hex())
                
                with pal_cols[i]:
                    border_style = "4px solid #667eea" if is_sel else "2px solid #aaa"
                    
                    st.markdown(
                        f'<div style="background:{hx}; height:45px; border-radius:8px; '
                        f'border:{border_style}; margin-bottom:8px;'
                        f'box-shadow:0 2px 5px rgba(0,0,0,0.15);"></div>',
                        unsafe_allow_html=True
                    )
                    
                    if st.button(hx.upper(), key=f"pal_btn_{i}", use_container_width=True):
                        _set_selected(rgb, f"Palette #{i+1}")
                        st.rerun()

        st.markdown("---")
        st.markdown("### 🔍 Custom Pixel Picker")
        st.caption(
            "Enter pixel coordinates (shown in most image viewers) to sample any colour."
        )

        h_img, w_img = arr.shape[:2]
        st.info(
            f"Image size: **{w_img} × {h_img}** px  —  "
            f"X ∈ [0, {w_img-1}]   Y ∈ [0, {h_img-1}]"
        )

        col_x, col_y, col_btn = st.columns([2, 2, 1])
        with col_x:
            px = st.number_input("X (column →)", 0, w_img - 1, w_img // 2)
        with col_y:
            py = st.number_input("Y (row ↓)",    0, h_img - 1, h_img // 2)
        with col_btn:
            st.write("")
            if st.button("🎯 Sample", width="stretch"):
                picked = get_pixel_color(arr, int(px), int(py))
                _set_selected(picked, f"Pixel ({int(px)}, {int(py)})")
                st.rerun()

        if st.session_state.selected_color:
            sel = st.session_state.selected_color
            sh  = rgb_to_hex(*sel)
            st.markdown(
                f'<div class="info-row">'
                f'<div class="dot-lg" style="background:{sh};"></div>'
                f'<div><strong>✅ Selected: {st.session_state.selected_color_name}</strong><br>'
                f'<code>{sh.upper()}</code> &nbsp;|&nbsp; '
                f'RGB({sel[0]}, {sel[1]}, {sel[2]})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    else:
        st.info("👆 Upload a photo or take one with your camera to begin.")
        st.markdown("""
        ---
        **What PaletteCraft does for you:**

        | Feature | Details |
        |---|---|
        | 🔬 Auto calibration | Removes lighting cast with Gray-World AWB |
        | 🎨 Palette extraction | K-Means clustering finds 5 dominant hues |
        | 🧪 Mixing recipes | Kubelka-Munk subtractive physics model |
        | 🗺️ Paint map | Neon overlay highlights where each colour lives |
        | 📦 Inventory | Manage your physical paint tubes |
        """)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — RECIPE BUILDER                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_recipe:
    st.markdown("### 🧪 Paint Recipe Builder")
    st.caption("💡 Select a colour in the **Image & Palette** tab. The recipe will generate automatically!")

    if not st.session_state.inventory:
        st.warning("⚠️ No paints in inventory — add some via the sidebar.")
    elif st.session_state.selected_color is None:
        st.info("👈 Select a colour in the **Image & Palette** tab first.")
    else:
        sel     = st.session_state.selected_color
        sel_hex = rgb_to_hex(*sel)

        col_tgt, col_opts = st.columns([1, 2])
        with col_tgt:
            st.markdown("**Target Colour**")
            st.markdown(
                f'{_color_block(sel_hex, 80, 80, 14)}'
                f'<br><code style="font-size:12px;">{sel_hex.upper()}</code>'
                f'<br><small>RGB({sel[0]}, {sel[1]}, {sel[2]})</small>',
                unsafe_allow_html=True,
            )
        with col_opts:
            st.markdown("**Options**")
            max_p = st.slider(
                "Maximum colours to mix", 1, 3, st.session_state.max_colors,
                help="Using 3 colours gives the best match but takes longer to calculate.",
                key="max_colors_slider"
            )
            if max_p != st.session_state.max_colors:
                st.session_state.max_colors = max_p
                if st.session_state.selected_color is not None:
                    with st.spinner("🔄 Recalculating recipe..."):
                        recipe = find_best_recipe(
                            sel,
                            st.session_state.inventory,
                            max_colors=max_p,
                        )
                        st.session_state.recipe = recipe
                        st.rerun()

            if st.button("🔄 Recalculate Recipe", type="primary", width="stretch"):
                with st.spinner("Optimising mixing ratios…"):
                    recipe = find_best_recipe(
                        sel,
                        st.session_state.inventory,
                        max_colors=st.session_state.max_colors,
                    )
                st.session_state.recipe = recipe
                st.rerun()

        if st.session_state.recipe:
            recipe  = st.session_state.recipe
            mixed   = recipe["mixed_rgb"]
            mix_hex = rgb_to_hex(*mixed)

            st.markdown("---")
            st.markdown("### 📋 Mixing Recipe")

            m1, m2, m3 = st.columns(3)
            m1.metric("Match Quality", recipe["match_quality"])
            m2.metric(
                "ΔE Colour Diff",
                f"{recipe['delta_e']:.1f}",
                help="0-5 Excellent · 5-10 Good · 10-20 Fair · >20 Poor",
            )
            m3.metric("Paints Used", len(recipe["paints"]))

            cmp1, cmp2 = st.columns(2)
            with cmp1:
                st.markdown("**🎯 Target**")
                st.markdown(
                    f'{_color_block(sel_hex, 64, 64, 8)}'
                    f'&nbsp; <code>{sel_hex.upper()}</code>',
                    unsafe_allow_html=True,
                )
            with cmp2:
                st.markdown("**🧪 Simulated Mix**")
                st.markdown(
                    f'{_color_block(mix_hex, 64, 64, 8)}'
                    f'&nbsp; <code>{mix_hex.upper()}</code>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.markdown("### 🖌️ Ingredients (Easy Scoop Method)")
            st.caption("👆 **1 part = 1 pea-sized blob** of paint on your palette.")

            total_parts = sum(p["parts"] for p in recipe["paints"])
            for paint in recipe["paints"]:
                pct = (paint["parts"] / total_parts * 100) if total_parts > 0 else 0
                scoops = round(paint["parts"] * 2) / 2
                
                col_dot, col_name, col_scoop, col_pct = st.columns([0.5, 2.5, 2, 3])
                with col_dot:
                    st.markdown(
                        f'<div style="width:22px;height:22px;background:{paint["hex"]};'
                        f'border-radius:50%;border:2px solid #666;margin-top:8px;"></div>',
                        unsafe_allow_html=True,
                    )
                with col_name:
                    st.write(f"**{paint['name']}**")
                with col_scoop:
                    st.write(f"**{scoops}** scoop{'s' if scoops != 1 else ''}")
                with col_pct:
                    st.write(f"{pct:.0f}% of mix")
                st.progress(pct / 100)

            summary = " + ".join(
                f"{round(p['parts'] * 2) / 2} scoop{'s' if round(p['parts'] * 2) / 2 != 1 else ''} of {p['name']}"
                for p in recipe["paints"]
            )
            st.markdown(
                f'<div class="recipe-block">📝 <strong>Mix this:</strong> {summary}</div>',
                unsafe_allow_html=True,
            )

            txt = (
                f"PaletteCraft — Paint Recipe\n"
                f"===========================\n"
                f"Target  : {sel_hex.upper()}  (RGB {sel[0]}, {sel[1]}, {sel[2]})\n"
                f"Quality : {recipe['match_quality']}\n"
                f"Delta-E : {recipe['delta_e']:.2f}\n\n"
                f"Ingredients (1 scoop = 1 pea-sized blob):\n"
            )
            for p in recipe["paints"]:
                scoops = round(p["parts"] * 2) / 2
                txt += f"  • {scoops} scoops of {p['name']}  ({p['hex'].upper()})\n"
            txt += f"\nSimulated mix: {mix_hex.upper()}\n"

            st.download_button(
                "💾 Download Recipe (.txt)",
                data=txt,
                file_name="palettecraft_recipe.txt",
                mime="text/plain",
                width="stretch",
            )

        else:
            st.info("💡 Select a colour to see the recipe here.")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — PAINT MAP                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
with tab_map:
    st.markdown("### 🗺️ Paint Map Visualiser")
    st.caption(
        "Highlights every region of the photo where your selected colour lives, "
        "so you know exactly where to apply each pigment."
    )

    if st.session_state.calibrated_array is None:
        st.info("Upload an image in the **Image & Palette** tab first.")
    elif st.session_state.selected_color is None:
        st.info("Select a target colour in the **Image & Palette** tab first.")
    else:
        sel     = st.session_state.selected_color
        sel_hex = rgb_to_hex(*sel)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            threshold = st.slider(
                "ΔE Threshold (match sensitivity)",
                2.0, 30.0, st.session_state.threshold, 0.5,
                help="Lower → only very close matches highlighted  |  Higher → wider colour range",
            )
            st.session_state.threshold = threshold
        with col_c2:
            dim = st.slider(
                "Background brightness",
                0.0, 1.0, 0.2, 0.05,
                help="0 = fully grayscale background  |  1 = fully coloured",
            )

        neon_name = st.selectbox("Highlight colour", list(NEON_COLOURS.keys()))
        neon_rgb  = NEON_COLOURS[neon_name]

        col_g, col_info = st.columns([1, 2])
        with col_g:
            gen = st.button("🗺️ Generate Paint Map", type="primary", width="stretch")
        with col_info:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-top:8px;">'
                f'<div style="width:20px;height:20px;background:{sel_hex};'
                f'border-radius:50%;border:1px solid #888;"></div>'
                f'<span>Target: <code>{sel_hex.upper()}</code> — '
                f'{st.session_state.selected_color_name}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        if gen:
            arr = st.session_state.calibrated_array
            with st.spinner("Building paint map…"):
                pm = generate_paint_map(
                    arr,
                    target_rgb=sel,
                    threshold=threshold,
                    overlay_color=neon_rgb,
                    overlay_alpha=0.65,
                    dim_factor=dim,
                )
                pct = get_match_percentage(arr, sel, threshold)
            st.session_state.paint_map_image = pm
            st.success(f"✅ ~{pct:.1f} % of image pixels match within ΔE < {threshold:.1f}")

        if st.session_state.paint_map_image:
            st.markdown("---")
            col_orig2, col_map2 = st.columns(2)
            with col_orig2:
                st.markdown("**Calibrated Image**")
                st.image(
                    Image.fromarray(st.session_state.calibrated_array),
                    width="stretch",
                )
            with col_map2:
                st.markdown("**🗺️ Paint Map Overlay**")
                st.image(st.session_state.paint_map_image, width="stretch")

            st.markdown(f"""
            **Legend:**
            - 🔳 Dimmed / greyscale → different from target colour
            - 🌸 **{neon_name}** highlight → areas matching target (ΔE ≤ {st.session_state.threshold:.1f})
            """)

            buf = io.BytesIO()
            st.session_state.paint_map_image.save(buf, format="PNG")
            st.download_button(
                "💾 Download Paint Map (.png)",
                data=buf.getvalue(),
                file_name="palettecraft_paint_map.png",
                mime="image/png",
                width="stretch",
            )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — MY INVENTORY                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
def is_valid_paint(p):
    return (
        isinstance(p, dict) and
        isinstance(p.get("name"), str) and
        isinstance(p.get("hex"), str) and
        isinstance(p.get("rgb"), list) and
        len(p["rgb"]) == 3 and
        all(isinstance(v, (int, float)) for v in p["rgb"])
    )

with tab_inv:
    st.markdown("### 🎨 My Paint Inventory")

    inv = st.session_state.inventory
    if not inv:
        st.warning("No paints yet — add them via the sidebar.")
    else:
        st.caption(f"{len(inv)} paint{'s' if len(inv) != 1 else ''} in your collection")

        CPR = 4
        for row_start in range(0, len(inv), CPR):
            cols = st.columns(CPR)
            for ci, col in enumerate(cols):
                idx = row_start + ci
                if idx < len(inv):
                    p = inv[idx]
                    with col:
                        st.markdown(
                            f'<div class="inv-card">'
                            f'<div style="width:52px;height:52px;background:{p["hex"]};'
                            f'border-radius:50%;border:2px solid #ccc;margin:0 auto;"></div>'
                            f'<div style="font-size:12px;font-weight:600;margin-top:6px;">'
                            f'{p["name"]}</div>'
                            f'<div style="font-size:10px;color:#888;font-family:monospace;">'
                            f'{p["hex"].upper()}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if st.button("🎯 Select", key=f"use_{p['id']}", width="stretch"):
                            _set_selected(p["rgb"], p["name"])
                            st.rerun()

        st.markdown("---")
        st.markdown("#### 💾 Export / Import Inventory")
        col_ex, col_im = st.columns(2)

        with col_ex:
            st.download_button(
                "📤 Export as JSON",
                data=json.dumps(inv, indent=2),
                file_name="palettecraft_inventory.json",
                mime="application/json",
                width="stretch",
            )

        with col_im:
            imp = st.file_uploader(
                "📥 Import JSON", type="json",
                key="imp_inv", label_visibility="collapsed",
            )
            if imp:
                try:
                    data = json.load(imp)
                    if isinstance(data, list) and all(is_valid_paint(p) for p in data):
                        for p in data:
                            if "id" not in p:
                                p["id"] = str(uuid.uuid4())[:8]
                        st.session_state.inventory = data
                        st.success(f"✅ Imported {len(data)} paints!")
                        st.rerun()
                    else:
                        st.error("Invalid format — expected a list of objects with valid {name, hex, rgb[3]}")
                except Exception as e:
                    st.error(f"Import error: {e}")


# ══════════════════════════════ FOOTER ════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#aaa;font-size:12px;">'
    "PaletteCraft · Kubelka-Munk subtractive mixing · CIELAB perceptual colour science"
    "</p>",
    unsafe_allow_html=True,
)