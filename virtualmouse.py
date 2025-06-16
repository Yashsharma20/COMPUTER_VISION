import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Screen size
screen_w, screen_h = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        continue

    # Flip and convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb_image)
    image_h, image_w, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get fingertip positions
            index_tip = hand_landmarks.landmark[8]  # Index finger tip
            thumb_tip = hand_landmarks.landmark[4]  # Thumb tip

            # Convert to pixel coordinates
            x = int(index_tip.x * image_w)
            y = int(index_tip.y * image_h)

            # Map to screen coordinates
            screen_x = np.interp(x, [0, image_w], [0, screen_w])
            screen_y = np.interp(y, [0, image_h], [0, screen_h])

            # Clamp to avoid fail-safe
            screen_x = max(1, min(screen_x, screen_w - 1))
            screen_y = max(1, min(screen_y, screen_h - 1))

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Detect pinch for click
            thumb_x = int(thumb_tip.x * image_w)
            thumb_y = int(thumb_tip.y * image_h)
            if abs(x - thumb_x) < 30 and abs(y - thumb_y) < 30:
                pyautogui.click()
                cv2.putText(image, 'Click!', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show image
    cv2.imshow("Virtual Mouse", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
