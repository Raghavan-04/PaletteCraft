# color_palette-fun
No problem at all! It is a highly complex application, so let’s break it down into plain English.

### 🎨 What is PaletteCraft (In Plain English)?
Imagine you are an artist. You see a beautiful sunset in a photo and want to paint it. 
Instead of **guessing** which colors to mix, you upload the photo to PaletteCraft. 
The app will:

1. Tell you the **5 main colors** in the photo.
2. Let you pick **any color** from the photo.
3. Look at your personal paint collection (your inventory).
4. **Calculate the exact recipe** (e.g., *"Mix 1.5 parts Cadmium Red + 0.8 parts Yellow Ochre"*) to physically recreate that color with real paint.

---

### ✨ Full List of Features (What the tabs do)

| Tab | Feature | What it does for you |
| :--- | :--- | :--- |
| **🖼️ Image & Palette** | **Auto Calibration** | Removes bad yellow/blue lighting from your photo (so colors look true-to-life). |
| | **Palette Extraction** | Uses AI (K-Means) to find the 5 most dominant colors in your photo. |
| | **Pixel Picker** | Click or type coordinates to sample *any specific pixel's* color. |
| **🧪 Recipe Builder** | **AI Mixing Engine** | Calculates the exact **parts** (e.g., 2.5 parts of this, 1 part of that) of your physical paints to match the target color. |
| | **Match Quality** | Tells you how close the mix is (Excellent / Good / Fair / Poor) using "Delta-E" (a scientific measure of color difference). |
| **🗺️ Paint Map** | **Heatmap Overlay** | Highlights every single area of your photo that matches your chosen color. The rest goes grey. It tells you exactly *where* to apply that paint. |
| **🎨 My Inventory** | **Paint Manager** | Add, remove, or import/export your physical paint tubes. The app uses this list to suggest recipes. |

---

### 🧠 Step-by-Step: How It Works (The Science simplified)

Here is exactly what happens when you click "Calculate Recipe":

**Step 1: You Upload a Photo**
- The app reads your image (JPG/PNG) into a big grid of pixels (RGB values: Red, Green, Blue).

**Step 2: "Auto White Balance" (Calibration)**
- *The Problem*: Photos taken under a lamp look yellow; under shade look blue.
- *The Fix*: The `calibration.py` script uses the **Gray-World assumption** (the average of all colors in nature is grey). It mathematically scales the red, green, and blue channels so that whites look pure white. This ensures the extracted colors are the *actual* paint colors, not the lighting color.

**Step 3: "Dominant Palette" (K-Means Clustering)**
- The app takes 8,000 random pixels from your image.
- It runs a Machine Learning algorithm (**K-Means**) that groups these pixels into 5 clusters based on their RGB similarity.
- The center of each cluster becomes one of your 5 palette colors. It sorts them by size (biggest blob of color first).

**Step 4: You Select a Target Color**
- You either click one of the 5 palette chips or pick a pixel manually. The app stores this as your `Target RGB`.

**Step 5: The "Recipe Builder" (The Heavy Math) - `mixing_engine.py`**
- *Step 5a*: It converts your target RGB into **CIELAB** space. 
  - *(Why? Because RGB is how screens glow. CIELAB is a math space designed to match how the **human eye** perceives color differences. A Delta-E of 1.0 is just barely noticeable to the human eye).*
- *Step 5b*: It looks at your inventory and finds the 7 closest paints to your target (to avoid testing 50 paints unnecessarily).
- *Step 5c*: It tries every combination of 1, 2, or 3 paints from that shortlist.
- *Step 5d*: For each combination, it uses **Kubelka-Munk Theory**. 
  - This is a physics formula that simulates how real **pigments absorb and scatter light** (unlike RGB, which just averages colors together). 
  - It uses a math optimizer (SLSQP) to find the exact percentage/weight of each paint to minimize the Delta-E (color difference).
- *Step 5e*: It converts those scientific weights into **painter-friendly "Parts"** (e.g., 1.5 parts, 0.25 parts) so you can physically measure it with a palette knife.

**Step 6: The "Paint Map" Visualisation**
- The app calculates the Delta-E (color difference) between your target color and *every single pixel* in the photo.
- If a pixel is within your set threshold (e.g., Delta-E < 8), it paints it **Neon Pink/Green**.
- If it is outside the threshold, it turns it to **greyscale**.
- This instantly shows you exactly where in the painting that specific mixed color should go.

---

### 🧑‍🎨 Summary (The 3 Golden Questions it answers)
1. **What colors are in this image?** → *Palette tab.*
2. **How do I mix that exact color with my paints?** → *Recipe tab.*
3. **Where do I put that color on the canvas?** → *Paint Map tab.*

It is basically a **GPS for paint mixing**—no more guessing, just pure math and physics to save you time and paint!

Now that you understand the flow, go ahead and upload a photo into your running app. Click "Calculate Recipe" and watch the magic happen. If the recipe comes back with a "Poor" match, it just means your inventory doesn't have the exact pigment needed (e.g., you might need to buy a specific blue to mix that shade). Let me know if you want me to explain any specific part deeper! 🚀