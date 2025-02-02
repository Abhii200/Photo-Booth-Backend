import cv2
import mediapipe as mp
import time
from escpos.printer import Usb

class GestureCaptureSystem:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Initialize printer (adjust vendor_id and product_id as per your printer)
        try:
            self.printer = Usb(0x0416, 0x5011)
            self.printer_available = True
        except:
            print("Printer not found. Images will be saved locally.")
            self.printer_available = False
            
    def detect_gesture(self, hand_landmarks):
        """Basic gesture detection based on finger positions"""
        if not hand_landmarks:
            return None
            
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Check for wave gesture (simplified)
        if (landmarks[8].y < landmarks[7].y and  # Index finger up
            landmarks[12].y < landmarks[11].y):   # Middle finger up
            return "wave"
            
        # Check for peace sign
        if (landmarks[8].y < landmarks[7].y and    # Index finger up
            landmarks[12].y < landmarks[11].y and   # Middle finger up
            landmarks[16].y > landmarks[15].y and   # Ring finger down
            landmarks[20].y > landmarks[19].y):     # Pinky down
            return "peace"
            
        return None
        
    def capture_and_print(self):
        prev_gesture = None
        gesture_start_time = 0
        
        while True:
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame")
                break
                
            # Keep a copy of the original frame without drawings
            original_frame = frame.copy()
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw hand landmarks on the display frame (not the one we'll save)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,  # Draw on the display frame
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Detect gesture
                    current_gesture = self.detect_gesture(hand_landmarks)
                    
                    # Handle gesture detection
                    if current_gesture:
                        if prev_gesture != current_gesture:
                            gesture_start_time = time.time()
                            prev_gesture = current_gesture
                        elif time.time() - gesture_start_time > 2:  # Hold gesture for 2 seconds
                            # Save original image without landmarks
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            filename = f"gesture_{current_gesture}_{timestamp}.jpg"
                            cv2.imwrite(filename, original_frame)  # Save the original frame
                            print(f"Captured {current_gesture} gesture! Saved as {filename}")
                            
                            # Print image if printer available
                            if self.printer_available:
                                try:
                                    self.printer.image(filename)
                                    self.printer.cut()
                                    print("Image sent to printer")
                                except Exception as e:
                                    print(f"Printing failed: {e}")
                            
                            prev_gesture = None
                            gesture_start_time = 0
            
            # Add text to show if a gesture is being held
            if prev_gesture:
                remaining_time = max(0, 2 - (time.time() - gesture_start_time))
                cv2.putText(frame, 
                           f"Hold {prev_gesture} for {remaining_time:.1f}s", 
                           (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2)
            
            # Display feed with landmarks and text
            cv2.imshow("Gesture Capture", frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
if __name__ == "__main__":
    print("Initializing Gesture Capture System...")
    print("Required packages:")
    print("- OpenCV (cv2)")
    print("- MediaPipe")
    print("- python-escpos")
    print("\nTo install required packages:")
    print("pip install opencv-python mediapipe python-escpos")
    print("\nInstructions:")
    print("1. Hold your hand in front of the camera")
    print("2. Make a gesture (wave or peace sign)")
    print("3. Hold the gesture for 2 seconds to capture")
    print("4. Press 'q' to quit")
    
    try:
        system = GestureCaptureSystem()
        system.capture_and_print()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        system.cleanup()