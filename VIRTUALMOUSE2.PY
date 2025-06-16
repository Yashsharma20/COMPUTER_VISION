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

# Flags to avoid repeated clicks
click_state = False
double_click_state = False

while True:
    success, image = cap.read()
    if not success:
        continue

    # Flip image and convert to RGB
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process with Mediapipe
    results = hands.process(rgb_image)
    image_h, image_w, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get key landmarks
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            middle_tip = hand_landmarks.landmark[12]

            # Convert to pixel positions
            x = int(index_tip.x * image_w)
            y = int(index_tip.y * image_h)
            thumb_x = int(thumb_tip.x * image_w)
            thumb_y = int(thumb_tip.y * image_h)
            middle_x = int(middle_tip.x * image_w)
            middle_y = int(middle_tip.y * image_h)

            # Move mouse using index finger
            screen_x = np.interp(x, [0, image_w], [0, screen_w])
            screen_y = np.interp(y, [0, image_h], [0, screen_h])
            screen_x = max(1, min(screen_x, screen_w - 1))
            screen_y = max(1, min(screen_y, screen_h - 1))
            pyautogui.moveTo(screen_x, screen_y)

            # --- Single Click (Thumb Opened) ---
            if thumb_tip.x < thumb_ip.x - 0.03:
                if not click_state:
                    pyautogui.click()
                    click_state = True
                    cv2.putText(image, 'Single Click (Thumb)', (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                click_state = False

            # --- Double Click (Thumb Touches Middle Finger) ---
            if abs(thumb_x - middle_x) < 30 and abs(thumb_y - middle_y) < 30:
                if not double_click_state:
                    pyautogui.doubleClick()
                    double_click_state = True
                    cv2.putText(image, 'Double Click (Thumb-Middle)', (x, y - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            else:
                double_click_state = False

    # Show the frame
    cv2.imshow("Virtual Mouse", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
