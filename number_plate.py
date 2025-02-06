import cv2
import os

# Constants
harcascade_xml = "model/haarcascade_russian_plate_number.xml"
capture_img = cv2.VideoCapture('Traffic Control CCTV.mp4')

capture_img.set(3, 640)  # width
capture_img.set(4, 480)  # height

min_area = 500
count = 0
resize_width = 320  # Desired width for the displayed image
resize_height = 240  # Desired height for the displayed image

# Create directory for saving images if it doesn't exist
output_dir = "detected_plate_numbers"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define margin size (in pixels)
margin = 20

while True:
    success, img = capture_img.read()
    if not success:
        break  # Exit loop if the video has ended or can't be read

    plate_cascade = cv2.CascadeClassifier(harcascade_xml)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_plate_numbers = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in detected_plate_numbers:
        area = w * h

        if area > min_area:
            # Calculate new coordinates with margin
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = x + w + margin
            y2 = y + h + margin

            # Ensure the new coordinates are within the image boundaries
            x1 = min(x1, img.shape[1])
            y1 = min(y1, img.shape[0])
            x2 = min(x2, img.shape[1])
            y2 = min(y2, img.shape[0])

            # Draw rectangle and add text
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Detected!", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            # Crop ROI with margin
            img_roi = img[y1: y2, x1: x2]
            cv2.imshow("ROI", img_roi)

            # Save detected plate image
            cv2.imwrite(f"{output_dir}/scanned_img_{count}.jpg", img_roi)
            count += 1

    # Resize image for display
    img_resized = cv2.resize(img, (resize_width, resize_height))
    cv2.imshow("Result", img_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

# Release the video capture object and close any open windows
capture_img.release()
cv2.destroyAllWindows()
