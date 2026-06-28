
# PaletteCraft 🎨 

**Paint Mixing Assistant for Artists**

PaletteCraft is an interactive Streamlit application that bridges the gap between digital photography and physical painting. By leveraging computer vision and perceptual color science, it analyzes your inspiration photos, extracts dominant color palettes, and calculates exact physical mixing recipes using the paints you actually own.

---

##  Features

* **Auto White-Balance Calibration:** Removes unnatural lighting casts from your photos using Gray-World and Max-RGB algorithms, ensuring you are color-matching the true pigments.
* **🎨 K-Means Palette Extraction:** Automatically identifies the 5 dominant hues in your image. Includes a Custom Pixel Picker for hyper-specific sampling.
* **🧪 Subtractive Recipe Engine:** Uses the **Kubelka-Munk** subtractive physics model to calculate exactly how to mix your physical paints to achieve a target color.
* **📏 Perceptual Color Matching:** Evaluates recipe accuracy using the **CIELAB $\Delta E_{76}$** standard, mimicking how the human eye perceives color differences.
* **🗺️ Interactive Paint Map:** Generates a visual neon overlay on your original photo, highlighting exactly where a specific mixed pigment should be applied on your canvas.
* **📦 Digital Paint Box:** Manage your physical inventory. Add custom paints via Hex codes, import/export your collection as JSON, and auto-calculate recipes based *only* on the tubes you own.

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/PaletteCraft.git](https://github.com/yourusername/PaletteCraft.git)
cd PaletteCraft

```

**2. Create a virtual environment (Recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

**3. Install dependencies**

```bash
pip install -r requirements.txt

```

*(Note: Ensure your `requirements.txt` includes `streamlit`, `Pillow`, and `numpy` at minimum, alongside any dependencies required by your `utils` folder).*

**4. Run the application**

```bash
streamlit run app.py

```

---

## 📂 Project Structure

```text
PaletteCraft/
├── app.py                  # Main Streamlit application UI and routing
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── utils/                  # Core math and computer vision backend
    ├── calibration.py      # White-balance algorithms
    ├── color_utils.py      # Hex/RGB/LAB conversions & Delta-E math
    ├── kmeans_palette.py   # Clustering and pixel sampling logic
    ├── mixing_engine.py    # Kubelka-Munk recipe optimization
    └── paint_map.py        # Image masking and visual overlay generation

```

---

## 📖 How to Use

The application is divided into four main tabs to guide your workflow:

1. **🖼️ Image & Palette:** Start here. Upload an inspiration photo or snap one with your webcam. The app will auto-calibrate the lighting and extract a clickable dominant color palette.
2. **🧪 Recipe Builder:** Once a color is selected, this tab reveals the exact mixing ratios. It breaks down the math into an "Easy Scoop Method" (e.g., 2 scoops Yellow + 0.5 scoops Crimson) and allows you to download a text file of the recipe for your studio.
3. **🗺️ Paint Map:** Visualizes where your newly mixed color belongs. Adjust the $\Delta E$ sensitivity threshold to see all the regions in the photo that match your target pigment.
4. **🎨 My Inventory:** Manage the paints you physically own. The recipe engine will *only* suggest mixes using active inventory items. Export your collection to keep it safe, or import new palettes via JSON.

---

## 🧠 The Science Behind the App

Unlike digital screens that mix light additively (RGB), physical paint absorbs light subtractively. Mixing `#FF0000` (Red) and `#00FF00` (Green) digitally yields Yellow, but mixing red and green physical paint yields a muddy brown.

PaletteCraft solves this by converting digital RGB values into perceptual **CIELAB** color space, and simulating physical pigment blending using a simplified **Kubelka-Munk** absorption/scattering model. This ensures the digital recipes translate accurately to your physical canvas.

---
