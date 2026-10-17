from playwright.sync_api import sync_playwright
import requests
import os
import time
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse, urlunparse

# ================= CONFIG =================
BASE_URL = r"https://www.pexels.com/search/squeezing%20lemon"
SAVE_FOLDER = r"D:\AI\CNN Classifier\FruitClassifier\Images\Train\Lemon"
MAX_IMAGES = 1000
TOTAL_PAGES = 100
# ================= DOWNLOAD =================
def download(url, path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers= headers,timeout=10)
        if "image" not in r.headers.get("Content-Type",""):
            print("Not an image: ",url)
            return False
        else:
            img = Image.open(BytesIO(r.content))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path,"JPEG")
            return True
    except Exception as e:
        print(e)
        return False

def saveLink(url):
    with open(r"D:\AI\CNN Classifier\FruitClassifier\Links\links.txt", "a") as f:
        f.write(url + "\n")
def saveCount(count):
    with open(r"D:\AI\CNN Classifier\FruitClassifier\Links\count.txt", "w") as f:
        f.write(str(count))
def loadCount():
    if not os.path.exists(r"D:\AI\CNN Classifier\FruitClassifier\Links\count.txt"):
        return 0
    with open(r"D:\AI\CNN Classifier\FruitClassifier\Links\count.txt", "r") as f:
        count = f.read().strip()
        return int(count) if count.isdigit() else 0
def loadLink():
    if not os.path.exists(r"D:\AI\CNN Classifier\FruitClassifier\Links\links.txt"):
        return set()
    with open(r"D:\AI\CNN Classifier\FruitClassifier\Links\links.txt","r") as f:
        return set(line.strip() for line in f)


def clearURL(url):
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path, 
                      '','',''))
# ================= SCRAPER =================
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()

    count = loadCount()
    seen = loadLink()

    for search in range(1):
        url = BASE_URL
        print(f"\nOpening: {url}")

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(2)

        for j in range(TOTAL_PAGES):
            
            images = page.locator("img[src*='images.pexels.com']")
            total = images.count()
            print("Images found:", total)

            for i in range(total):
                img = images.nth(i)

                try:
                    src = (
                            img.get_attribute("src") 
                            or img.get_attribute("data-src") 
                            or img.get_attribute("data-srcset")
                            )
                except:
                    continue

                if not src:
                    continue
                if src.startswith("//"):
                    src = "https:"+src
                if "/user/" in src or "_96x96" in src:
                    continue

                if src in seen:
                    print("Seen!")
                    continue
                

                save_path = os.path.join(SAVE_FOLDER, f"img_{count}.jpg")

                success = download(src, save_path)

                if success:
                    seen.add(src)
                    saveLink(src)
                    print(f"[{count}] Saved:", src)
                    saveCount(count)
                    count += 1
                else:
                    print("Skipped:", src)
            
            page.mouse.wheel(0, 1000)
            time.sleep(3)      
            button = page.get_by_role("button",name="Load more")
            if button.count() > 0:
                try:
                    btn = button.first
                    btn.scroll_into_view_if_needed()
                    btn.click(timeout=8000)
                    print("Clicked load more")
                    time.sleep(2)
                except Exception as e:
                    print(e)
            else:
                print("No load button found!")
                break

    print(f"\nDONE. Downloaded {count} images.")
    browser.close()