import streamlit as st
import json
import os
from pathlib import Path

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

# ---------- Basic Info ----------
st.title(f"🛰️ Base ID: {selected_base['id']} ({selected_base['country']})")
st.write(f"**Coordinates:** {selected_base['latitude']}, {selected_base['longitude']}")

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

lat = selected_base['latitude']
lon = selected_base['longitude']

st.map([{"lat": lat, "lon": lon}])

google_maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
st.markdown(f"[🌐 Open in Google Maps]({google_maps_link})")
