"""
base_analyzer.py – step 1: grab Google Earth screenshots
--------------------------------------------------------
Usage:
    python base_analyzer.py grab-shots
"""
import json
import os
import time
import pathlib
from dataclasses import dataclass
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image
from io import BytesIO
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import multiprocessing
import requests

# ---------- configurable constants ----------
CSV_PATH           = "military_bases.csv"
START_INDEX        = 2 
ROWS_TO_PROCESS    = 4  # number of rows to process from CSV
# – Camera defaults (tweak to taste) –
ALTITUDE_M         = 8000                 # metres above sea level (…a)
DISTANCE           = 8000                 # camera distance (…d)
TILT_DEG           = 0                    # 0 = nadir (straight down)
HEADING_DEG        = 0                    # 0 = face north
ROLL_DEG           = 0                    # leave 0

ZOOM_IN_FACTOR     = 0.6
ZOOM_OUT_FACTOR    = 1.4
LON_SHIFT          = 0.01                 # in degrees longitude

LOAD_TIMEOUT       = 15                   # seconds to wait for tiles

SCREEN_DIR = pathlib.Path("screenshots")
SCREEN_DIR.mkdir(exist_ok=True)

# RESPONSE_DIR = pathlib.Path("responses")
# RESPONSE_DIR.mkdir(exist_ok=True)

DATA_JSON_PATH = pathlib.Path("data.json")

STABILITY_CHECK_INTERVAL = 2  # seconds between comparisons
STABILITY_THRESHOLD = 1       # % similarity to count as "stable"
STABILITY_ROUNDS = 2          # how many rounds of similarity we require
# --------------------------------------------

# Load Gemini and OpenRouter api keys from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
openrouter_key = os.getenv("OPENROUTER_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash-lite")
OPENROUTER_MODEL = "microsoft/mai-ds-r1:free"

@dataclass
class Base:
    id: int
    country: str
    lat: float
    lon: float

def read_first_bases(csv_path: str, n_rows: int, start_index: int = 0) -> list[Base]:
    df = pd.read_csv(csv_path, skiprows=range(1, start_index + 1), nrows=n_rows)
    df.columns = ["id", "country", "latitude", "longitude", "google_maps_link"]
    bases = [
        Base(
            id=row["id"],
            country=row["country"],
            lat=row["latitude"],
            lon=row["longitude"],
        )
        for _, row in df.iterrows()
    ]
    return bases

def _gemini_worker(img_path, country, queue, allow_zoom_in, history_of_analysts):
    try:
        if allow_zoom_in:
            prompt = (
                f"You are an expert in understanding satellite imagery and you work for the US army. "
                f"We got intel that this area is a base/facility of the military of {country}. Analyze this image and "
                f"respond ONLY with a JSON (DO NOT write the word 'json') object containing the following keys:\n"
                f"1. 'findings': A list of findings that you think are important for the US army to know, including "
                f"all man-made structures, military equipment, and infrastructure. We are trying to find which "
                f"systems, weapons, or equipment are present so focus on that.\n"
                f"2. 'analysis': A detailed analysis of your findings.\n"
                f"3. 'things_to_continue_analyzing': A list of things that you think are important to continue "
                f"analyzing in further images.\n"
                f"4. 'action': One of ['zoom-in', 'zoom-out', 'move-left', 'move-right', 'finish'] based on what "
                f"would help you analyze the image or area better.\n"
                f"- Choose 'zoom-in' if you need to zoom in the image\n"
                f"- Choose 'zoom-out' if you need more context of the surrounding area or if you are zoomed "
                f"in too much\n"
                f"- Choose 'move-left' or 'move-right' if you suspect there are important features just outside "
                f"the current view, such as related buildings, storage facilities, etc..\n"
                f"Also if you can only see forest or image with nothing interesting in it, consider moving left/right/zooming out\n"
                f"- Choose 'finish' if you have a complete understanding of the location\n"
                f"DO NOT include coordinates in your response\n"
            )
        else:
            prompt = (
                f"You are an expert in understanding satellite imagery and you work for the US army. "
                f"We got intel that this area is a base/facility of the military of {country}. Analyze this image and "
                f"respond ONLY with a JSON (DO NOT write the word 'json') object containing the following keys:\n"
                f"1. 'findings': A list of findings that you think are important for the US army to know, including "
                f"all man-made structures, military equipment, and infrastructure. We are trying to find which "
                f"systems, weapons, or equipment are present so focus on that.\n"
                f"2. 'analysis': A detailed analysis of your findings.\n"
                f"3. 'things_to_continue_analyzing': A list of things that you think are important to continue "
                f"analyzing in further images.\n"
                f"4. 'action': One of ['move-left', 'move-right', 'finish', 'zoom-out'] based on what "
                f"would help you analyze the image or area better.\n"
                f"- Choose 'zoom-out' if you need more context of the surrounding area or if you are zoomed "
                f"in too much\n"
                f"- Choose 'move-left' or 'move-right' if you suspect there are important features just outside "
                f"the current view, such as related buildings, storage facilities, etc.. "
                f"Also if you can only see forest or image with nothing interesting in it, consider moving left/right/zooming out\n"
                f"- Choose 'finish' if you have a complete understanding of the location\n"
                f"DO NOT include coordinates in your response\n"
            )
        analysis_addition = f"Here is the analysis of previous analysts about this area and their recommendations. You can use this data but don’t use it as fact, think for yourself: {history_of_analysts}"
        prompt = prompt if not history_of_analysts else prompt + analysis_addition
        img = Image.open(img_path)
        response = model.generate_content([prompt, img])
        queue.put(response.text)
    except Exception as e:
        queue.put(f"[ERROR] Gemini API failed: {e}")

def analyze_with_gemini(img_path: str, country: str, allow_zoom_in: bool, history_of_analysts: dict, timeout_seconds: int = 30):

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_gemini_worker, args=(img_path, country, queue, allow_zoom_in, history_of_analysts))
    proc.start()
    proc.join(timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return "[TIMEOUT] Gemini API call took too long and was forcefully stopped."

    return queue.get() if not queue.empty() else "[ERROR] No response from Gemini."

def new_driver() -> webdriver.Chrome:
    """Launch Chrome with GUI (not headless)."""
    opts = Options()
    opts.add_argument("--log-level=3")  # Errors only (1=INFO, 2=WARNING, 3=ERROR)
    # opts.add_argument("--start-maximized")
    service = Service(ChromeDriverManager().install())  
    driver = webdriver.Chrome(service=service, options=opts)  
    # Set to half screen: 960x1080 (adjust to your needs)
    driver.set_window_size(960, 1080)
    driver.set_window_position(0, 0)  # top-left corner     
    return driver

def ask_commander_model(messages, model="microsoft/mai-ds-r1:free"):
    headers = {
        "Authorization": f"Bearer {openrouter_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/amitfld/base_analyzer",  # optional but recommended
        "X-Title": "military-commander-analysis"
    }

    data = {
        "model": model,
        "messages": messages
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


def get_screenshot_image(driver):
    png = driver.get_screenshot_as_png()
    return Image.open(BytesIO(png)).convert("RGB")

def is_visually_blank(image: Image.Image, threshold: float = 0.9) -> bool:
    """
    Check if the image is mostly a single color (e.g. Earth loading screen).
    Returns True if 'blank'.
    """
    arr = np.array(image)
    pixels = arr.reshape(-1, 3)  # flatten to list of RGB
    dominant_color, counts = np.unique(pixels, axis=0, return_counts=True)
    top_color_ratio = np.max(counts) / pixels.shape[0]
    return top_color_ratio >= threshold


def image_similarity(img1, img2):
    arr1 = np.asarray(img1).astype("float")
    arr2 = np.asarray(img2).astype("float")

    if arr1.shape != arr2.shape:
        return 0.0

    mse = np.mean((arr1 - arr2) ** 2)
    max_val = 255.0
    return 1.0 - (mse / (max_val ** 2))  # similarity: 1.0 = identical

def wait_for_tiles(driver: webdriver.Chrome):
    """
    Wait for Google Earth to fully load, fly to the location,
    and visually stabilize the imagery.
    """
    print("⏳ Waiting for Earth UI to load...")
    WebDriverWait(driver, LOAD_TIMEOUT).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    print("⏳ Waiting for camera to reach target...")
    target_url = driver.current_url
    WebDriverWait(driver, LOAD_TIMEOUT).until(
        lambda d: d.current_url == target_url
    )

    print("⏳ Waiting for imagery to stabilize...")
    prev_img = get_screenshot_image(driver)

    for _ in range(20):  # Max wait = 40 seconds
        time.sleep(STABILITY_CHECK_INTERVAL)
        new_img = get_screenshot_image(driver)

        sim = image_similarity(prev_img, new_img)
        print(f"🔍 Image similarity: {sim:.4f}")

        if is_visually_blank(new_img):
            print("⚠️ Image appears blank or stalled. Waiting and retrying...")
            prev_img = new_img  # update anyway to detect future changes
            continue

        if sim >= STABILITY_THRESHOLD:
            print("✅ Imagery stabilized.")
            return new_img
        prev_img = new_img

    print("⚠️ Timeout: imagery did not fully stabilize.")
    return prev_img


def extract_clean_json(raw_text: str) -> str:
    """
    Extracts the portion of a string that starts with the first '{' and ends with the last '}'.
    Useful for cleaning LLM output like ```json ... ``` wrappers.
    """
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1 and start < end:
        return raw_text[start:end+1]
    return raw_text  # fallback: return unchanged if malformed

def build_earth_url(lat, lon, distance_m):
    return (
        f"https://earth.google.com/web/@"
        f"{lat},{lon},{ALTITUDE_M:.2f}a,"
        f"{distance_m:.2f}d,{TILT_DEG:.2f}y,"
        f"{HEADING_DEG:.0f}h,0t,{ROLL_DEG:.0f}r"
    )

def grab_screens():
    # 📁 Load existing data.json (if exists), otherwise start fresh
    if DATA_JSON_PATH.exists():
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            data_json = json.load(f)
    else:
        data_json = {}

    # 📁 Load ALL bases starting from START_INDEX
    all_bases = read_first_bases(CSV_PATH, n_rows=None, start_index=START_INDEX)

    # 🔍 Filter to only new (not yet analyzed)
    new_bases = [b for b in all_bases if str(b.id) not in data_json]

    # 🚫 If no new bases left
    if not new_bases:
        print("✅ All bases are already analyzed. Nothing to do.")
        return

    # Limit to ROWS_TO_PROCESS
    bases_to_process = new_bases[:ROWS_TO_PROCESS]
    driver = new_driver()

    for i, base in enumerate(bases_to_process, 1):
        # 🔍 Skip if this base was already analyzed (based on base.id as string key)
        if str(base.id) in data_json:
            print(f"🚫 Skipping base {base.id} ({base.country}) — already analyzed in data.json.")
            continue

        # Initialize values
        current_distance = DISTANCE
        current_lat = base.lat
        current_lon = base.lon
        zoom_in_count = 0
        allow_zoom_in = True
        history_of_analysts = {}

        print(f"\nProcessing base {i} of {ROWS_TO_PROCESS} -> Base id: {base.id} located in {base.country}")

        for step in range(1, 9):  # up to 8 AI "analysts"

            # 1. Generate Earth URL with current camera params
            url = build_earth_url(current_lat, current_lon, current_distance)
            print(f"\n📸 [Step {step}/8] Capturing image at distance: {current_distance:.0f}, lon: {current_lon:.5f}\nURL = {url}")

            # 2. Open it in Selenium, wait, capture screenshot
            driver.get(url)
            stable_img = wait_for_tiles(driver)

            # Resize and save image
            w_target = 1024
            w_orig, h_orig = stable_img.size
            scale = w_target / w_orig
            new_size = (w_target, int(h_orig * scale))
            resized_img = stable_img.resize(new_size, Image.LANCZOS)

            country_clean = base.country.lower().replace(" ", "_")
            jpeg_path = SCREEN_DIR / f"{base.id}_{country_clean}_step{step}.jpg"
            resized_img.save(jpeg_path, format="JPEG", quality=90)
            print(f"    ↳ saved → {jpeg_path }")

            # Decide if allow zoom in
            if zoom_in_count > 2:
                allow_zoom_in = False

            # 🧠 Analyze with Gemini
            print("🧠 Sending image to Gemini for analysis...")
            result = analyze_with_gemini(jpeg_path , base.country, allow_zoom_in, history_of_analysts)

            # ⏳ Optional retry if it timed out
            if "TIMEOUT" in result:
                print("⚠️ Retrying Gemini call once after timeout...")
                result = analyze_with_gemini(jpeg_path, base.country, allow_zoom_in, history_of_analysts)                    

            # 💾 Save response to text file
            cleaned_result = extract_clean_json(result)
            # history_of_analysts[f"analysis number {step}"] = cleaned_result
            try:
                # Parse JSON string into a Python dict for structured storage
                parsed_result = json.loads(cleaned_result)
            except Exception as e:
                print(f"⚠️ Failed to parse JSON for analyst {step}, saving as raw string. ({e})")
                parsed_result = cleaned_result  # fallback to raw text if parsing fails

            history_of_analysts[f"analysis number {step}"] = parsed_result

            # 5. Parse action and update camera state
            if cleaned_result.strip().startswith("{"):
                try:
                    action = json.loads(cleaned_result)["action"]
                    print(f"🎯 Gemini suggested action: {action}")
                except Exception as e:
                    print(f"❌ Failed to extract 'action'. Ending analysis. ({e})")
                    break
            else:
                print("❌ Gemini response was not JSON. Ending analysis.")
                print(f"Response content:\n{cleaned_result}")
                break

            if action == "zoom-in":
                current_distance *= ZOOM_IN_FACTOR
                zoom_in_count += 1
            elif action == "zoom-out":
                current_distance *= ZOOM_OUT_FACTOR
            elif action == "move-left":
                current_lon -= LON_SHIFT
            elif action == "move-right":
                current_lon += LON_SHIFT
            elif action == "finish":
                print("✅ Gemini marked analysis as complete.")
                break
            else:
                print("⚠️ Unrecognized action. Ending analysis.")
                break
        
        # 6. Final report
        print("\n📝 Generating final report...")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a military commander reviewing satellite analysis reports from 8 analysts. "
                    "Each analyst examined a different satellite image of the same suspected enemy base. "
                    "You must synthesize their findings into a final intelligence report for decision-makers. "
                    "👉 IMPORTANT: Respond ONLY with a JSON (DO NOT write the word 'json') object containing the following keys:\n"
                    "- 'summary': A summary of key findings (with estimated confidence)\n"
                    "- 'strategic_analysis': Analysis of enemy capabilities and threats\n"
                    "- 'conflicting_opinions': Conflicting points between analysts and how you resolved them\n"
                    "- 'final_recommendation': One of ['Continue surveillance', 'Deploy recon drones', 'Launch strike', 'Archive', 'Other']\n"
                    "- 'justification': Detailed justification for your recommendation\n"
                    "DO NOT include any text outside the JSON object."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here is what the analysts reported:\n\n{history_of_analysts}\n\n"
                    "Write the final report as per the instructions above."
                )
            }
        ]

        commander_output = ask_commander_model(messages, model=OPENROUTER_MODEL)

        # Clean the JSON if needed
        cleaned_commander = extract_clean_json(commander_output)

        try:
            # Parse JSON string into Python dict
            parsed_commander_report = json.loads(cleaned_commander)
        except Exception as e:
            print(f"⚠️ Failed to parse commander report JSON, saving as raw string. ({e})")
            parsed_commander_report = cleaned_commander  # fallback to raw text if parsing fails

        # 💾 Save everything to data.json
        data_to_save = {
            "id": base.id,
            "country": base.country,
            "latitude": base.lat,
            "longitude": base.lon,
            "analysts": history_of_analysts,
            "commander_report": parsed_commander_report
        }

        data_json[str(base.id)] = data_to_save

        # 💾 Save updated data.json immediately
        with open(DATA_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data_json, f, indent=2, ensure_ascii=False)

        print(f"✅ All data saved to {DATA_JSON_PATH}")
    driver.quit()
    print("\n🎉 All tasks completed.")


# ---------- CLI entry ----------
if __name__ == "__main__":
    grab_screens()

