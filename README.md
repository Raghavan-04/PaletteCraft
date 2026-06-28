
# PaletteCraft 🎨 

**Paint Mixing Assistant for Artists**

PaletteCraft is an interactive Streamlit application that bridges the gap between digital photography and physical painting. By leveraging computer vision and perceptual color science, it analyzes your inspiration photos, extracts dominant color palettes, and calculates exact physical mixing recipes using the paints you actually own.

---

##  Features

* **Auto White-Balance Calibration:** Removes unnatural lighting casts from your photos using Gray-World and Max-RGB algorithms, ensuring you are color-matching the true pigments.
* **K-Means Palette Extraction:** Automatically identifies the 5 dominant hues in your image. Includes a Custom Pixel Picker for hyper-specific sampling.
* **Subtractive Recipe Engine:** Uses the **Kubelka-Munk** subtractive physics model to calculate exactly how to mix your physical paints to achieve a target color.
* **Perceptual Color Matching:** Evaluates recipe accuracy using the **CIELAB $\Delta E_{76}$** standard, mimicking how the human eye perceives color differences.
* **Interactive Paint Map:** Generates a visual neon overlay on your original photo, highlighting exactly where a specific mixed pigment should be applied on your canvas.
* **Digital Paint Box:** Manage your physical inventory. Add custom paints via Hex codes, import/export your collection as JSON, and auto-calculate recipes based *only* on the tubes you own.

##  Why PaletteCraft is Different

Most color-picking apps on the market fail artists because they treat digital pixels and physical paint as the same thing. PaletteCraft is built specifically for the realities of the traditional studio:

* **It matches the object, not the lighting:** If you take a photo of a white coffee cup under a warm tungsten bulb, most apps will tell you to paint the cup orange. PaletteCraft's auto-calibration strips away ambient lighting casts so you mix the true local color of the object.
* **It knows physical paint behaves badly:** Standard apps use digital RGB/CMYK math. They assume mixing blue and yellow creates perfect green. In reality, physical pigments rely on complex absorption and scattering. PaletteCraft uses actual optical physics (Kubelka-Munk) to predict these muddy, non-linear physical interactions.
* **It works with what you have:** There is no point in an app telling you to use "Cobalt Teal" if you don't own it. PaletteCraft's inventory system restricts its mathematical solver to only use the physical tubes currently sitting on your desk.

---

## Kubelka-Munk Theory

Unlike digital screens that mix light additively (RGB), physical paint absorbs light subtractively. To accurately simulate this, PaletteCraft utilizes the **Kubelka-Munk Theory of Reflectance**, a foundational mathematical model used in the industrial paint, textile, and paper industries since 1931.

Instead of simply averaging RGB values, the Kubelka-Munk model evaluates paints based on two distinct optical properties:
* **$K$ (Absorption):** How much light the pigment absorbs (turning it into heat).
* **$S$ (Scattering):** How much light the pigment scatters back to the viewer's eye.

The relationship between a paint layer's reflectance ($R$) and its absorption and scattering is defined by the core Kubelka-Munk equation:

$$\frac{K}{S} = \frac{(1 - R)^2}{2R}$$

To further study for different application: 
- [Colorimetric Analysis of Sweat for Sodium Monitoring](https://github.com/Raghavan-04/Colorimetric-Analysis-of-Sweat-for-Sodium-monitoring)
  
### How PaletteCraft Uses It:
When you ask the app to mix a recipe, it doesn't just average the hex codes. 
1. It converts your target color into its theoretical reflectance.
2. It calculates the individual $\frac{K}{S}$ ratios for the paints in your inventory.
3. It simulates mixing them at various concentration ratios ($c_1, c_2, ...$). Because physical paint mixing is non-linear (adding 10% black darkens a mix far more than adding 10% white lightens it), this physics-based approach ensures the resulting recipe matches how real paint behaves on a real palette.
4. Finally, it measures the simulated physical mix against your target color using CIELAB perceptual space ($\Delta E$) to find the absolute best match.
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
