import streamlit as st
import json
import os
from pathlib import Path
import folium
from streamlit_folium import st_folium
from fpdf import FPDF


# ---------- Setup ----------
DATA_PATH = Path("data.json")
SCREENSHOT_DIR = Path("screenshots")

# ---------- Load Data ----------
if DATA_PATH.exists():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    st.error("❌ data.json file not found.")
    st.stop()

# ---------- Sidebar ----------
base_ids = list(data.keys())
base_selection = st.sidebar.selectbox(
    "Select a base to view:",
    options=base_ids,
    format_func=lambda k: f"{k} - {data[k]['country']}"
)

selected_base = data[base_selection]

# ---------- Quick Stats: Threat Levels ----------
threat_levels = {"High": 0, "Medium": 0, "Low": 0, "Unknown": 0}

for base in data.values():
    commander_report = base.get("commander_report", {})
    strategic_data = commander_report.get("strategic_analysis", {})

    # Determine threats: if strategic_data is a dict, look for keys; if it's a string, treat as direct threat text
    if isinstance(strategic_data, dict):
        threats = strategic_data.get("threats") or strategic_data.get("threat_assessment")
    elif isinstance(strategic_data, str):
        threats = strategic_data  # treat full string as threat text
    else:
        threats = ""

    # Normalize to a string
    threat_text = ""
    if isinstance(threats, list):
        threat_text = " ".join(threats).lower()
    elif isinstance(threats, str):
        threat_text = threats.lower()

    if not threat_text.strip():
        threat_levels["Unknown"] += 1
    elif any(keyword in threat_text for keyword in ["missile", "strike", "offensive", "sam", "attack", "combat"]):
        threat_levels["High"] += 1
    elif any(keyword in threat_text for keyword in ["patrol", "defense", "interdiction", "asymmetric", "monitor"]):
        threat_levels["Medium"] += 1
    else:
        threat_levels["Low"] += 1

st.sidebar.markdown("**🚨 Threat Levels Overview:**")
for level, count in threat_levels.items():
    st.sidebar.markdown(f"- {level}: {count}")

# ---------- Basic Info ----------
st.title(f"🛰️ Base ID: {selected_base['id']} ({selected_base['country']})")
st.write(f"**Coordinates:** {selected_base['latitude']}, {selected_base['longitude']}")

# Google Maps link
lat = selected_base['latitude']
lon = selected_base['longitude']
google_maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
st.markdown(f"[🌐 Open in Google Maps]({google_maps_link})")


# ---------- Final Screenshot ----------
country_clean = selected_base['country'].lower().replace(" ", "_")
last_step = max(
    [int(k.split()[-1]) for k in selected_base['analysts'].keys()]
)

screenshot_filename = f"{selected_base['id']}_{country_clean}_step{last_step}.jpg"
screenshot_path = SCREENSHOT_DIR / screenshot_filename

st.subheader("🖼️ Final Satellite Image")
if screenshot_path.exists():
    st.image(screenshot_path, caption=f"Step {last_step} - Final image", use_container_width=True)
else:
    st.warning(f"Screenshot not found: {screenshot_filename}")
# ---------- Commander Report ----------
st.header("📝 Commander Final Report")

commander_report = selected_base.get("commander_report", None)

if not commander_report:
    st.warning("No commander report found.")
else:
    # 1️⃣ Summary
    summary_data = commander_report.get("summary", {})
    st.subheader("🔍 Summary")

    if isinstance(summary_data, dict):
        for key, val in summary_data.items():
            key_name = key.replace('_', ' ').capitalize()
            st.markdown(f"**{key_name}:**")
            
            if isinstance(val, list):
                # Original behavior: list of findings
                for item in val:
                    st.markdown(f"- {item}")
            
            elif isinstance(val, dict):
                # 🆕 New: handle dictionary of findings (key-value)
                for inner_key, inner_val in val.items():
                    inner_key_name = inner_key.replace('_', ' ').capitalize()
                    st.markdown(f"- **{inner_key_name}:** {inner_val}")
            
            else:
                # String fallback
                st.markdown(val)

    else:
        # If summary_data is a string
        st.markdown(summary_data)


    # 2️⃣ Strategic Analysis
    strategic_data = commander_report.get("strategic_analysis", {})
    st.subheader("📊 Strategic Analysis")
    if isinstance(strategic_data, dict):
        for key, val in strategic_data.items():
            key_name = key.replace('_', ' ').capitalize()
            st.markdown(f"**{key_name}:**")
            if isinstance(val, list):
                for item in val:
                    st.markdown(f"- {item}")
            else:
                st.markdown(f"{val}")
    else:
        # If strategic_data is a string
        st.markdown(strategic_data)

    # 3️⃣ Conflicting Opinions
    conflicts = commander_report.get("conflicting_opinions", {})
    st.subheader("⚠️ Conflicting Opinions")

    if conflicts:
        if isinstance(conflicts, dict):
            for key, val in conflicts.items():
                key_name = key.replace('_', ' ').capitalize()
                st.markdown(f"**{key_name}:**")
                if isinstance(val, list):
                    for item in val:
                        st.markdown(f"- {item}")
                else:
                    st.markdown(f"{val}")
        else:
            # If conflicts is not a dict (edge case)
            st.markdown(conflicts)
    else:
        st.markdown("_No conflicting opinions reported._")

    # 4️⃣ Final Recommendation
    reco = commander_report.get("final_recommendation", None)
    st.subheader("✅ Final Recommendation")
    if reco:
        st.markdown(f"**{reco}**")
    else:
        st.markdown("_No recommendation provided._")

    # 5️⃣ Justification
    justification = commander_report.get("justification", None)
    st.subheader("🛠️ Justification")
    if justification:
        for paragraph in justification.split("\n\n"):
            st.markdown(paragraph.strip())
    else:
        st.markdown("_No justification provided._")


st.subheader("📦 Export Report")

# ---------- Export buttons ----------
col1, col2, col3 = st.columns(3)

# 1️⃣ Export as JSON
with col1:
    json_data = json.dumps(commander_report, indent=4)
    st.download_button(
        label="🗂️ Download JSON",
        data=json_data,
        file_name=f"base_{selected_base['id']}_report.json",
        mime="application/json"
    )

# 2️⃣ Export as PDF
with col2:
    if st.button("📄 Export as PDF"):
        # Save PDF file locally
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(200, 10, txt=f"Commander Report - Base {selected_base['id']}", ln=True, align="C")
        pdf.ln(10)

        def add_section(title, content):
            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 10, txt=title)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, txt=content)
            pdf.ln(5)

        # Add all sections
        add_section("Summary", json.dumps(commander_report.get("summary", ""), indent=2))
        add_section("Strategic Analysis", json.dumps(commander_report.get("strategic_analysis", ""), indent=2))
        add_section("Conflicting Opinions", json.dumps(commander_report.get("conflicting_opinions", ""), indent=2))
        add_section("Final Recommendation", commander_report.get("final_recommendation", ""))
        add_section("Justification", commander_report.get("justification", ""))

        output_pdf_path = f"base_{selected_base['id']}_report.pdf"
        pdf.output(output_pdf_path)

        with open(output_pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF",
                data=f,
                file_name=output_pdf_path,
                mime="application/pdf"
            )

# 3️⃣ Export Image
with col3:
    if screenshot_path.exists():
        with open(screenshot_path, "rb") as img_file:
            st.download_button(
                label="🖼️ Download Final Image",
                data=img_file,
                file_name=screenshot_filename,
                mime="image/jpeg"
            )
    else:
        st.warning("⚠️ No final image available for download.")


# ---------- Full History ----------
with st.expander("🔎 View Full Analysis History"):
    for step_num in range(1, last_step + 1):
        analysis_key = f"analysis number {step_num}"
        analysis_data = selected_base['analysts'].get(analysis_key)
        
        st.markdown(f"### 📝 Analyst {step_num}")

        # Show Screenshot
        step_screenshot_filename = f"{selected_base['id']}_{country_clean}_step{step_num}.jpg"
        step_screenshot_path = SCREENSHOT_DIR / step_screenshot_filename

        if step_screenshot_path.exists():
            st.image(step_screenshot_path, caption=f"Step {step_num}", use_container_width=True)
        else:
            st.warning(f"Screenshot not found: {step_screenshot_filename}")

        # Show Findings + Action
        if analysis_data:
            try:
                if isinstance(analysis_data, str):
                    analysis_json = json.loads(analysis_data)
                else:
                    analysis_json = analysis_data  # already a dict
                st.markdown("**Findings:**")
                st.markdown("- " + "\n- ".join(analysis_json.get("findings", [])))
                
                st.markdown("**Analysis:**")
                st.markdown(analysis_json.get("analysis", "(No analysis)"))
                
                st.markdown("**Things to Continue Analyzing:**")
                st.markdown("- " + "\n- ".join(analysis_json.get("things_to_continue_analyzing", [])))
                
                st.markdown(f"**Action Taken:** `{analysis_json.get('action', 'N/A')}`")
            except json.JSONDecodeError:
                st.warning("Could not parse JSON for this step.")
        else:
            st.info("No analysis data for this step.")


# ---------- Map View ----------
st.subheader("🗺️ Map View")

# Create base map centered around selected base
m = folium.Map(location=[selected_base['latitude'], selected_base['longitude']],
               zoom_start=5,
               tiles='OpenStreetMap')  

# Add all bases as smaller pins
for base_id, base in data.items():
    folium.Marker(
        [base['latitude'], base['longitude']],
        popup=f"Base {base['id']} - {base['country']}",
        tooltip=f"Base {base['id']}",
        icon=folium.Icon(color="green", icon="glyphicon glyphicon-flag")
    ).add_to(m)

# Add marker for the current base (highlighted)
folium.Marker(
    [selected_base['latitude'], selected_base['longitude']],
    popup=f"Base {selected_base['id']} - {selected_base['country']}",
    tooltip=f"📍 Selected Base: Base {selected_base['id']}",
    icon=folium.Icon(color="green", icon="glyphicon glyphicon-screenshot")
).add_to(m)

# Render it in Streamlit
st_data = st_folium(m, width=700, height=500)
