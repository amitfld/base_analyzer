# 🛰️ Military Base OSINT Analyzer

This project is an **AI-powered OSINT (Open Source Intelligence) analyzer** that automates the collection, analysis, and reporting of **military bases** using **satellite imagery from Google Earth**. It combines **LLM-based satellite image analysis**, **data synthesis**, and an interactive **Streamlit GUI** for exploring insights.

---

## 🚀 Project Overview

The system simulates a **team of AI analysts** that review satellite images of military facilities across multiple countries. Each analyst reviews a specific image, suggests follow-up actions (like zooming or moving the camera), and passes their findings forward. After multiple iterations, a **commander model** aggregates all the insights and provides a **final intelligence report**.

This project was developed as part of the **"Idea to Reality" academic course** and uses a combination of:

- **Google Earth + Selenium** for automated image grabbing
- **Google Generative AI (Gemini API)** and **OpenRouter LLMs** for satellite image analysis
- **Streamlit + Folium** for interactive exploration of the analyzed data

---

## 🧩 Components

### 1️⃣ `base_analyzer.py`

This script **automates the entire analysis pipeline**:

- Loads military base coordinates from `military_bases.csv`
- Opens Google Earth in Selenium and **takes high-res screenshots**
- Sends images to Gemini LLM with a **detailed prompt for military analysis**
- Lets the analyst decide next actions: `zoom-in`, `zoom-out`, `move-left`, `move-right`, or `finish`
- Loops through up to **8 steps per base**, allowing the AI team to refine the analysis iteratively
- Collects and saves **analyst reports** + a **commander final report** into `data.json`
- Skips already-analyzed bases for **incremental progress**

Key features:
- Dynamic camera adjustments (zoom & pan)
- Stability checks to ensure Earth images fully load
- JSON extraction & cleanup for structured AI responses

---

### 2️⃣ `base_info_app.py`

An **interactive dashboard** built with **Streamlit** that:

- Lets users **select a base** from the analyzed list
- Displays the **final satellite image**
- Shows a **summary, strategic analysis, conflicting opinions, and final recommendation**
- **Exports reports** as:
  - 📄 PDF
  - 🗂️ JSON
  - 🖼️ Image
- Provides an interactive **map view** with Folium marking all bases
- Includes a **threat level overview** sidebar and a **full step-by-step analyst history**

---

## 🛠️ Tech Stack & Libraries

- **Google Earth & Selenium**: Automates satellite image capture
- **Google Generative AI (Gemini API)**: Vision-based military analysis
- **OpenRouter API**: Final report synthesis via advanced LLMs
- **Pandas**: Handles CSV data
- **Pillow (PIL)**: Image processing
- **NumPy**: Image similarity checks
- **Streamlit + Folium**: GUI and interactive map
- **FPDF**: PDF report export

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

---

## 📦 How It Works

### 🔍 Analyst Loop

For each military base:

1. **Load base data** (ID, country, coordinates)
2. **Open Google Earth** → Fly to location
3. **Capture screenshot** (scaled to 1024px width, JPEG format)
4. **Send image to Gemini LLM**
5. **Receive JSON**:
    ```json
    {
      "findings": [...],
      "analysis": "...",
      "things_to_continue_analyzing": [...],
      "action": "zoom-in"
    }
    ```
6. **Adjust camera based on `action`** (zoom or pan)
7. Repeat steps 3–6 for up to 8 iterations

### 🪖 Commander Report

Once all analysts finish:

- Collates all analyst reports
- Sends to the commander LLM on OpenRouter
- Receives a final JSON report with:
    - `summary`
    - `strategic_analysis`
    - `conflicting_opinions`
    - `final_recommendation`
    - `justification`

### 💾 Data Storage

All data is stored in `data.json`, structured like:

```json
{
  "base_id": {
    "id": ...,
    "country": ...,
    "latitude": ...,
    "longitude": ...,
    "analysts": {
      "analysis number 1": {...},
      ...
    },
    "commander_report": {...}
  }
}
```

---

## 🖥️ Usage

### 1️⃣ Analyze Bases

```bash
python base_analyzer.py
```
- 🚨 Make sure you have a valid `.env` file with your API keys:
    ```
    GEMINI_API_KEY=your_gemini_api_key
    OPENROUTER_API_KEY=your_openrouter_api_key
    ```

### 2️⃣ Explore Data

```bash
streamlit run base_info_app.py
```

---

## 📸 Screenshots & Features

- ✅ Final satellite image of each base
- ✅ Step-by-step analyst screenshots + findings
- ✅ Commander’s final report (strategy & recommendation)
- ✅ Downloadable reports (PDF, JSON)
- ✅ Interactive map of all analyzed bases

---

## ✨ Extra Notes

- The camera **stabilization logic** ensures Earth tiles fully load before screenshots are taken
- **Timeouts & retries** are handled to manage LLM API limits and network hiccups
- The app supports **incremental runs**: if `data.json` already contains a base, it skips it automatically
- **Threat detection** uses a simple NLP heuristic in the Streamlit app to classify threat levels as High/Medium/Low

---

## ✅ TODO / Ideas for Expansion

- 📊 Add historical image analysis (Google Earth’s time slider)
- 🚀 Launch parallel Selenium sessions for faster processing
- 🏙️ Use GPS reverse-geocoding to fetch nearest cities for context
- ✈️ Integrate OpenSky API to check airspace activity near each base
- 🔍 Use Folium to draw bounding boxes or detected features on maps

---

## 👤 Author

Made with ❤️ for the **"Idea to Reality" 2025 course.**

GitHub: [@amitfld](https://github.com/amitfld)

---

