import cv2
from ultralytics import YOLO
import pygame
import os

# 1. Initialize Sound
pygame.mixer.init()
mp3_file = "alert1.mp3" 
alert_sound = pygame.mixer.Sound(mp3_file) if os.path.exists(mp3_file) else None

# 2. Load Model & Camera
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

# 3. Calibration
MACHINE_LINE_Y = 70
SAFETY_LINE_Y = 130

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    
    # 4. Run Detection
    results = model(frame, conf=0.25)

    # Draw Safety Lines
    cv2.line(frame, (0, SAFETY_LINE_Y), (frame.shape[1], SAFETY_LINE_Y), (0, 255, 255), 2)
    cv2.line(frame, (0, MACHINE_LINE_Y), (frame.shape[1], MACHINE_LINE_Y), (0, 0, 255), 3)

    for r in results:
        # Get the names dictionary from the model (e.g., {0: 'HAND'})
        names = r.names 
        
        for box in r.boxes:
            # A. Get Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # B. Get Class ID and Name
            cls_id = int(box.cls[0])
            label_name = names[cls_id] # This will be 'HAND'
            
            # C. Get Confidence Score
            conf = float(box.conf[0])
            
            # D. Create the display text (e.g., "HAND 0.85")
            display_text = f"{label_name} {conf:.2f}"

            # E. DRAW EVERYTHING
            # Draw the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Draw the label and confidence background
            cv2.putText(frame, display_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # --- SAFETY LOGIC ---
            if y1 < SAFETY_LINE_Y and y1 > MACHINE_LINE_Y:
                if alert_sound and not pygame.mixer.get_busy():
                    alert_sound.play()

            if y1 <= MACHINE_LINE_Y:
                print("EMERGENCY STOP")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("Shredder Safety Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()