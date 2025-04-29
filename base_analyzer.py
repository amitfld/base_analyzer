"""
base_analyzer.py – step 1: grab Google Earth screenshots
--------------------------------------------------------
Usage:
    python base_analyzer.py grab-shots
"""
import os
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

# ---------- configurable constants ----------
CSV_PATH           = "military_bases.csv"
ROWS_TO_PROCESS    = 5                     # ← change later if you like
# – Camera defaults (tweak to taste) –
ALTITUDE_M         = 8000                 # metres above sea level (…a)
DISTANCE_M         = 9000                 # camera distance (…d)
TILT_DEG           = 0                    # 0 = nadir (straight down)
HEADING_DEG        = 0                    # 0 = face north
ROLL_DEG           = 0                    # leave 0
LOAD_TIMEOUT       = 15                   # seconds to wait for tiles

SCREEN_DIR = pathlib.Path("screenshots")
SCREEN_DIR.mkdir(exist_ok=True)

STABILITY_CHECK_INTERVAL = 2  # seconds between comparisons
STABILITY_THRESHOLD = 1       # % similarity to count as "stable"
STABILITY_ROUNDS = 2          # how many rounds of similarity we require
# --------------------------------------------


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


def read_first_bases(csv_path: str, n_rows: int) -> list[Base]:
    df = pd.read_csv(csv_path, nrows=n_rows)
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
        if sim >= STABILITY_THRESHOLD:
            print("✅ Imagery stabilized.")
            return new_img
        prev_img = new_img

    print("⚠️ Timeout: imagery did not fully stabilize.")
    return prev_img



def grab_screens():
    bases = read_first_bases(CSV_PATH, ROWS_TO_PROCESS)
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


        screenshot_path = SCREEN_DIR / f"{base.id}_{base.country.lower().replace(' ', '_')}.png"
        resized_img.save(screenshot_path)
        print(f"    ↳ saved → {screenshot_path}")

    driver.quit()


# ---------- CLI entry ----------
if __name__ == "__main__":
    grab_screens()

