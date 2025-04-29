"""
base_analyzer.py – step 1: grab Google Earth screenshots
--------------------------------------------------------
Usage:
    python base_analyzer.py grab-shots
"""
import os
import sys
import time
import pathlib
from dataclasses import dataclass
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from PIL import Image
from io import BytesIO
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import concurrent.futures
import multiprocessing

# ---------- configurable constants ----------
CSV_PATH           = "military_bases.csv"
START_INDEX = 0
ROWS_TO_PROCESS    = 5
# – Camera defaults (tweak to taste) –
ALTITUDE_M         = 8000                 # metres above sea level (…a)
DISTANCE_M         = 18000                # camera distance (…d)
TILT_DEG           = 0                    # 0 = nadir (straight down)
HEADING_DEG        = 0                    # 0 = face north
ROLL_DEG           = 0                    # leave 0
LOAD_TIMEOUT       = 15                   # seconds to wait for tiles

SCREEN_DIR = pathlib.Path("screenshots")
SCREEN_DIR.mkdir(exist_ok=True)

RESPONSE_DIR = pathlib.Path("responses")
RESPONSE_DIR.mkdir(exist_ok=True)

STABILITY_CHECK_INTERVAL = 2  # seconds between comparisons
STABILITY_THRESHOLD = 1       # % similarity to count as "stable"
STABILITY_ROUNDS = 2          # how many rounds of similarity we require
# --------------------------------------------

# Load Gemini api key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash-lite")

@dataclass
class Base:
    id: int
    country: str
    lat: float
    lon: float

    def earth_url(self) -> str:
        """Build Google-Earth Web URL with our camera parameters."""
        return (f"https://earth.google.com/web/@"
                f"{self.lat},{self.lon},{ALTITUDE_M:.2f}a,"
                f"{DISTANCE_M:.2f}d,{TILT_DEG:.2f}y,"
                f"{HEADING_DEG:.0f}h,0t,{ROLL_DEG:.0f}r")


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

def _gemini_worker(img_path, country, queue):
    try:
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
            f"the current view\n"
            f"- Choose 'finish' if you have a complete understanding of the location"
        )
        img = Image.open(img_path)
        response = model.generate_content([prompt, img])
        queue.put(response.text)
    except Exception as e:
        queue.put(f"[ERROR] Gemini API failed: {e}")

def analyze_with_gemini(img_path: str, country: str, timeout_seconds: int = 30):

    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_gemini_worker, args=(img_path, country, queue))
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

def grab_screens():
    bases = read_first_bases(CSV_PATH, ROWS_TO_PROCESS, START_INDEX)
    driver = new_driver()

    for i, base in enumerate(bases, 1):
        url = base.earth_url()
        print(f"\n[{i}/{len(bases)}] {base.id} – opening {url}")
        driver.get(url)
        stable_img = wait_for_tiles(driver)

        # Resize to width=1024, maintain aspect ratio
        w_target = 1024
        w_orig, h_orig = stable_img.size
        scale = w_target / w_orig
        new_size = (w_target, int(h_orig * scale))
        resized_img = stable_img.resize(new_size, Image.LANCZOS)

        country_clean = base.country.lower().replace(" ", "_")
        jpeg_path = SCREEN_DIR / f"{base.id}_{country_clean}.jpg"
        resized_img.save(jpeg_path, format="JPEG", quality=90)
        print(f"    ↳ saved → {jpeg_path }")

        # 🧠 Analyze with Gemini
        print("🧠 Sending image to Gemini for analysis...")
        result = analyze_with_gemini(jpeg_path , base.country)

        # ⏳ Optional retry if it timed out
        if "TIMEOUT" in result:
            print("⚠️ Retrying Gemini call once after timeout...")
            result = analyze_with_gemini(jpeg_path, base.country, timeout_seconds=30)

        # 💾 Save response to text file
        cleaned_result = extract_clean_json(result)
        response_path = RESPONSE_DIR / f"{base.id}_{country_clean}_response.json"
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(cleaned_result)
        print(f"    ↳ Gemini response saved → {response_path}")
        print("\n📄 Gemini analysis:\n" + cleaned_result)

    driver.quit()


# ---------- CLI entry ----------
if __name__ == "__main__":
    grab_screens()

