# Import required libraries
import cv2
import mediapipe as mp
import numpy as np
import IPython.display as display
from IPython.display import HTML, Image
import time

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class HandGestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
    def detect_symbol(self, landmarks):
        """Detect the gesture symbol based on finger positions"""
        # Define finger landmarks
        finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
        finger_bases = [5, 9, 13, 17]  # Corresponding bases
        thumb_tip = 4
        thumb_base = 2
        
        # Convert landmarks to numpy array for easier processing
        landmarks_array = np.array([[l.x, l.y, l.z] for l in landmarks])
        
        # Check if finger is raised (y coordinate comparison)
        def is_finger_raised(tip_idx, base_idx):
            return landmarks_array[tip_idx][1] < landmarks_array[base_idx][1]
        
        # Check if thumb is raised (x coordinate comparison)
        def is_thumb_raised():
            return landmarks_array[thumb_tip][0] < landmarks_array[thumb_base][0]
        
        # Check fingers state
        raised_fingers = [is_finger_raised(tip, base) 
                         for tip, base in zip(finger_tips, finger_bases)]
        thumb_raised = is_thumb_raised()
        
        # Determine gesture
        if not thumb_raised and not any(raised_fingers):
            return "DANGER", (0, 0, 255)  # Red for danger
        elif all(raised_fingers) and thumb_raised:
            return "SAFE", (0, 255, 0)    # Green for safe
        return "UNDEFINED", (128, 128, 128)  # Gray for undefined

    def process_frame(self, frame):
        """Process a single frame and return the annotated image"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(frame_rgb)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Detect gesture
                symbol, color = self.detect_symbol(hand_landmarks.landmark)
                
                # Add text to frame
                cv2.putText(
                    frame,
                    symbol,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA
                )
                
        return frame

def run_detection(duration=30):
    """Run the hand gesture detection for a specified duration"""
    detector = HandGestureDetector()
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    start_time = time.time()
    
    try:
        while (time.time() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process the frame
            processed_frame = detector.process_frame(frame)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            display.clear_output(wait=True)
            display.display(Image(data=cv2.imencode('.jpg', frame_rgb)[1].tobytes()))
            
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Example usage in notebook:
# Run the detection for 30 seconds
# run_detection(30)
