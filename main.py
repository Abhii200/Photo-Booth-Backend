from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import time
from PIL import Image
import os
import base64
from io import BytesIO
from datetime import datetime
import platform

app = FastAPI()

# Enable CORS
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://photo-booth-seven-murex.vercel.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:5173"],
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
    "frame": None,
    "pending_image": None,
    "pending_image_path": None
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

def get_frame_base64():
    """Convert the current frame to base64"""
    if capture_session["frame"] is None:
        return None
    _, buffer = cv2.imencode('.jpg', capture_session["frame"])
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/approve-image")
async def approve_image(data: dict):
    """Handle image approval or rejection"""
    try:
        approved = data.get("approved", False)
        
        if capture_session["pending_image_path"] is None:
            raise HTTPException(status_code=400, detail="No pending image")

        if approved:
            # Image is already saved, just need to handle printing if available
            if platform.system() == "Windows":
                try:
                    # Add your printing logic here if needed
                    pass
                except Exception as e:
                    print(f"Printing error: {e}")
        else:
            # Delete the rejected image
            if os.path.exists(capture_session["pending_image_path"]):
                os.remove(capture_session["pending_image_path"])

        # Reset pending image state
        capture_session["pending_image"] = None
        capture_session["pending_image_path"] = None
        capture_session["wave_detected"] = False
        capture_session["status"] = "Waiting for wave gesture..."

        return {"status": "Image processed successfully"}
    except Exception as e:
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
        return sorted(images, key=lambda x: x["timestamp"], reverse=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start-capture")
async def start_capture():
    global cap, capture_session
    if capture_session["active"]:
        raise HTTPException(status_code=400, detail="Capture session already active")

    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        capture_session.update({
            "active": True,
            "countdown": None,
            "status": "Waiting for wave gesture...",
            "wave_detected": False,
            "countdown_start": 0,
            "frame": None,
            "pending_image": None,
            "pending_image_path": None
        })
        return {"status": "Capture session started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop-capture")
async def stop_capture():
    global cap, capture_session
    if not capture_session["active"]:
        raise HTTPException(status_code=400, detail="No active capture session")

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
        # If there's a pending image, return it without capturing new frames
        if capture_session["pending_image"]:
            return {
                "active": True,
                "countdown": None,
                "status": "Please confirm or reject the photo",
                "frame": None,
                "pendingImage": capture_session["pending_image"]
            }

        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to grab frame")

        frame = cv2.flip(frame, 1)
        capture_session["frame"] = frame
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
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"photo_{timestamp}.jpg"
                filepath = os.path.join("captured_images", filename)
                cv2.imwrite(filepath, frame)

                # Create the pending image data
                with open(filepath, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode()
                
                capture_session["pending_image"] = {
                    "id": filename,
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "timestamp": timestamp
                }
                capture_session["pending_image_path"] = filepath
                capture_session["status"] = "Photo taken! Please confirm or reject."
                capture_session["countdown"] = None
                
                return {
                    "active": True,
                    "countdown": None,
                    "status": "Photo taken! Please confirm or reject.",
                    "frame": None,
                    "pendingImage": capture_session["pending_image"]
                }

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

@app.on_event("startup")
async def startup_event():
    """Create necessary directories on startup"""
    os.makedirs("captured_images", exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    global cap
    if cap and cap.isOpened():
        cap.release()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)