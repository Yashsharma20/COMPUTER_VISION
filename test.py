import cv2

# Use webcam (0 or 1 depending on device)
cap = cv2.VideoCapture(0)  # Adjust index if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
