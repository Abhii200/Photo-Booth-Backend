from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import time
from PIL import Image, ImageWin
import os
import base64
from io import BytesIO
from datetime import datetime
import win32print
import win32ui
import win32con

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure captured_images directory exists
os.makedirs("captured_images", exist_ok=True)

# Global state
capture_session = {
    "active": False,
    "countdown": None,
    "status": "",
    "wave_detected": False,
    "countdown_start": 0,
    "frame": None  # Store the latest frame
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Initialize camera
cap = None

def print_image(image_path):
    """Print image directly to default printer"""
    try:
        # Open image
        image = Image.open(image_path)

        # Get default printer
        printer_name = win32print.GetDefaultPrinter()

        # Open printer
        hprinter = win32print.OpenPrinter(printer_name)

        # Create device context
        hdc = win32ui.CreateDC()
        hdc.CreatePrinterDC(printer_name)

        # Configure printer settings
        printer_info = win32print.GetPrinter(hprinter, 2)
        devmode = printer_info['pDevMode']
        devmode.BitsPerPel = 24  # Color printing
        devmode.PelsWidth = image.size[0]
        devmode.PelsHeight = image.size[1]
        devmode.Orientation = 1  # Portrait

        # Start print job
        hdc.StartDoc('Photo Booth Image')
        hdc.StartPage()

        # Convert image to Windows-compatible DIB
        dib = ImageWin.Dib(image)

        # Get printer dimensions
        printer_width = hdc.GetDeviceCaps(win32con.PHYSICALWIDTH)
        printer_height = hdc.GetDeviceCaps(win32con.PHYSICALHEIGHT)

        # Scale image to fit printer page while maintaining aspect ratio
        image_ratio = image.size[0] / image.size[1]
        printer_ratio = printer_width / printer_height

        if image_ratio > printer_ratio:
            scaled_width = printer_width
            scaled_height = int(printer_width / image_ratio)
        else:
            scaled_height = printer_height
            scaled_width = int(printer_height * image_ratio)

        # Center the image on the page
        x = int((printer_width - scaled_width) / 2)
        y = int((printer_height - scaled_height) / 2)

        # Print the image
        dib.draw(hdc.GetHandleOutput(), (x, y, x + scaled_width, y + scaled_height))

        # End print job
        hdc.EndPage()
        hdc.EndDoc()

        # Clean up
        hdc.DeleteDC()
        win32print.ClosePrinter(hprinter)

        return True
    except Exception as e:
        print(f"Printing error: {e}")
        return False

def get_frame_base64():
    """Convert the current frame to base64"""
    if capture_session["frame"] is None:
        return None
    _, buffer = cv2.imencode('.jpg', capture_session["frame"])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/start-capture")
async def start_capture():
    global cap, capture_session

    if capture_session["active"]:
        raise HTTPException(status_code=400, detail="Capture session already active")

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        capture_session = {
            "active": True,
            "countdown": None,
            "status": "Waiting for wave gesture...",
            "wave_detected": False,
            "countdown_start": 0,
            "frame": None
        }
        return {"status": "Capture session started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-capture")
async def stop_capture():
    global cap, capture_session

    if not capture_session["active"]:
        raise HTTPException(status_code=400, detail="No active capture session to stop")

    try:
        capture_session["active"] = False
        if cap:
            cap.release()
        return {"status": "Capture session stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capture-status")
async def get_capture_status():
    global cap, capture_session

    if not capture_session["active"]:
        return {
            "active": False,
            "countdown": None,
            "status": "No active session",
            "frame": None
        }

    try:
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to grab frame")

        frame = cv2.flip(frame, 1)
        capture_session["frame"] = frame  # Store the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y

                if (wrist_y - index_tip_y > 0.15) and (wrist_y - middle_tip_y > 0.15):
                    if not capture_session["wave_detected"]:
                        capture_session["wave_detected"] = True
                        capture_session["countdown_start"] = time.time()
                        capture_session["status"] = "Wave detected! Starting countdown..."

        if capture_session["wave_detected"]:
            elapsed = time.time() - capture_session["countdown_start"]
            remaining = max(3 - int(elapsed), 0)
            capture_session["countdown"] = remaining

            if remaining == 0:
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"captured_images/photo_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                # Print the image
                if print_image(filename):
                    capture_session["status"] = "Image captured and printed!"
                else:
                    capture_session["status"] = "Image captured but printing failed"

                # Reset the wave detection for continuous capture
                capture_session["wave_detected"] = False
                capture_session["countdown"] = None
                capture_session["status"] = "Waiting for wave gesture..."

        return {
            "active": True,
            "countdown": capture_session["countdown"],
            "status": capture_session["status"],
            "frame": get_frame_base64()
        }

    except Exception as e:
        capture_session["active"] = False
        if cap:
            cap.release()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images")
async def get_images():
    """Get all captured images"""
    try:
        images = []
        if not os.path.exists("captured_images"):
            return images

        for filename in os.listdir("captured_images"):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                with open(os.path.join("captured_images", filename), "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode()
                    timestamp = filename.split("_")[1].split(".")[0]
                    images.append({
                        "id": filename,
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "timestamp": timestamp
                    })
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    os.makedirs("captured_images", exist_ok=True)
    # print(f"Using printer: {win32print.GetDefaultPrinter()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global cap
    if cap and cap.isOpened():
        cap.release()
