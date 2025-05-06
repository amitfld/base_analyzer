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
- The system then uses a **commander LLM** to combine all analyst reports into a **final intelligence report** with strategic insights and a clear recommendation.
- Collects and saves **analyst reports** + a **commander final report** into `data.json`
- Skips already-analyzed bases for **incremental progress**

Key features:
- Automated Satellite Imagery Capture
- Dynamic camera adjustments (zoom & pan)
- Multi-step AI Analysis
- Persistent Progress Tracking

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

## 🖥️ How To Run

Follow these steps to set up and run the project:

1️⃣ **Clone the Repository**

```bash
git clone https://github.com/amitfld/base_analyzer.git
cd base_analyzer
```

2️⃣ **Set Up a Virtual Environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3️⃣ **Install Requirements**

```bash
pip install -r requirements.txt
```

4️⃣ **Get API Keys**

- 🔑 **Gemini API Key:** [Get your Gemini API key here](https://aistudio.google.com/prompts/new_chat?pli=1)
- 🔑 **OpenRouter API Key:** [Get your OpenRouter API key here](https://openrouter.ai/)

5️⃣ **Create `.env` File**

In the project root, create a file named `.env` and add your API keys:

```
GEMINI_API_KEY=your_gemini_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

6️⃣ **Configure Script Settings**

Open `base_analyzer.py` and adjust the following variables to match your needs:

```python
START_INDEX = 0               # Index of the base to start from
ROWS_TO_PROCESS = 5           # Number of bases to process
```

7️⃣ **Run the Analyzer**

```bash
python base_analyzer.py
```

This will:

- Analyze each base by taking screenshots and sending them to the AI analysts
- Save the full results (analyst reports + commander report) to `data.json`
- Store images in the `screenshots/` folder

8️⃣ **Explore the Data**

Launch the Streamlit app to interactively explore the reports:

```bash
streamlit run base_info_app.py
```


✅ **That’s it!** You can now browse all military base reports, download PDFs, view maps, and more.

---

## 🖼️📄 Screenshots & Example Results

Result examples generated by the app and downloaded can be found in [`results_examples/`](./results_examples) folder.

App screenshots can be found in [`results_examples/app_screenshots/`](./results_examples/app_screenshots) folder. This 

Feel free to explore them directly in the repository!

---

## 📸 Screenshots & Features

- ✅ Final satellite image of each base
- ✅ Step-by-step analyst screenshots + findings
- ✅ Commander’s final report (strategy & recommendation)
- ✅ Downloadable reports (PDF, JSON)
- ✅ Interactive map of all analyzed bases

---

## ✅ TODO / Ideas for Expansion

- 📊 Add historical image analysis (Google Earth’s time slider)
- 🚀 Launch parallel Selenium sessions for faster processing
- 🏙️ Use GPS reverse-geocoding to fetch nearest cities for context
- ✈️ Integrate OpenSky API to check airspace activity near each base
- 🔍 Use Folium to draw bounding boxes or detected features on maps

---

## ⚖️ Legal & Ethical Notice

This project strictly adheres to **legal and ethical guidelines**:

- ✅ All images are sourced from **publicly accessible Google Earth** using its **standard web interface**.
- ✅ The analysis is performed **only on publicly available satellite imagery** and **does not involve hacking, intrusion, or unauthorized access** of any kind.
- ✅ No sensitive, confidential, or classified information is used or processed.
- ✅ The purpose of this project is **academic and educational**—to demonstrate how **AI can assist in OSINT (Open Source Intelligence)** using legal data sources.

➡️ This project **does not engage in or promote** any illegal surveillance or military intelligence operations. It is aligned with **fair use principles** for research and learning.

---

## 🌐 Data Source

The military base data in `military_bases.csv` was sourced from **OSINT Military Base Map** for educational and research purposes.

🔗 [Visit the original data source](https://sites.google.com/view/osintmilitarymap/)

---

## 🙌 Acknowledgments
This project is part of the From Idea to Reality Using AI course. Thanks to the instructor for guidance and to the open-source community for the incredible libraries that made this possible.

---

## 👤 Author

Made with ❤️ for the **"Idea to Reality" 2025 course.**

GitHub: [@amitfld](https://github.com/amitfld)

LinkdIn: [Amit Feldman](https://www.linkedin.com/in/amit-fld/)

---

