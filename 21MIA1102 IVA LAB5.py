import cv2
import numpy as np
import os 
import dlib
import csv
# Load the video from the specified path
image_path = 'videofile.mp4'
cap = cv2.VideoCapture(image_path)

# Output folder for gender-detected images
output_folder = r'C:\Users\KIIT\Desktop\College\IVA'
os.makedirs(output_folder, exist_ok=True)

# Load the image
image = cv2.imread(image_path)
# Get video frame width, height, and FPS for saving the output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object to save the output video in the specified location
output_path = r'C:/Users/KIIT/Desktop/College/IVA/frames_extracted/output_motion_detection.avi'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

# Frame differencing
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
event_frames = []  # List to store event timestamps

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for comparison
    current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Frame differencing
    diff_frame = cv2.absdiff(prev_frame_gray, current_frame_gray)

    # Apply threshold to isolate regions with significant motion
    _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)

    # Find contours of moving areas
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Event detection based on contour size (significant motion)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust threshold for "significant" motion
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Annotate the frame
            cv2.putText(frame, "Event Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            event_frames.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)  # Store event time in seconds

    # Write the frame with highlighted motion to the output video
    out.write(frame)

    # Update the previous frame
    prev_frame_gray = current_frame_gray.copy()

# Release video objects
cap.release()
out.release()

# Output event timestamps
print(f"Event timestamps (in seconds): {event_frames}")


# --- Step 1: Preprocessing: Skin-color-based detection to detect faces and hands ---
# Convert the image to HSV for skin color detection
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define skin color range in HSV
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Threshold the image to get only skin-colored regions
skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

# Perform morphological operations to remove small noise
kernel = np.ones((3, 3), np.uint8)
skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)

# Bitwise-AND mask and original image to extract the skin regions
skin = cv2.bitwise_and(image, image, mask=skin_mask)

# Save the preprocessed image with skin detection
cv2.imwrite(f'{output_folder}\Task2-preprocessing.jpg', skin)

# --- Step 2: Gesture Analysis - Detect facial features using dlib ---
# Load the facial landmark predictor from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this file is in the correct path
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Convert image to grayscale for facial landmark detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray_image)

detected_emotions = []  # List to store emotions of each individual

# Loop through each detected face and find facial landmarks
for face in faces:
    landmarks = predictor(gray_image, face)

    # Draw the facial landmarks on the face
    for n in range(36, 48):  # Eyes region
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

    for n in range(48, 68):  # Mouth region
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # --- Step 3: Emotion Classification ---
    # Calculate the curvature of the mouth (points 48-54 for the upper lip, 55-61 for lower)
    mouth_up = landmarks.part(51).y  # Upper lip point
    mouth_down = landmarks.part(57).y  # Lower lip point

    # Classify emotion based on mouth curvature
    if mouth_up < mouth_down:  # Smiling if upper lip is higher than lower lip
        emotion = "Happy"
    else:  # Frowning or neutral
        emotion = "Sad"
    
    # Draw the emotion label on the image
    cv2.putText(image, emotion, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Append the detected emotion to the list
    detected_emotions.append(emotion)

# Save the output with facial landmarks and classified emotions
cv2.imwrite(f'{output_folder}\Task2-gesture_analysis.jpg', image)

# --- Step 4: Categorize Overall Sentiment ---
# Count the occurrences of each emotion
happy_count = detected_emotions.count("Happy")
sad_count = detected_emotions.count("Sad")

# Determine the majority sentiment
if happy_count > sad_count:
    overall_sentiment = "Majority Happy"
elif sad_count > happy_count:
    overall_sentiment = "Majority Sad"
else:
    overall_sentiment = "Neutral Crowd"

# Save the overall sentiment to a CSV file
with open(f'{output_folder}\Task2-overall_sentiment.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Happy Count', 'Sad Count', 'Overall Sentiment'])
    writer.writerow([happy_count, sad_count, overall_sentiment])

# Output the final results
print("Image Processing and Sentiment Analysis Complete!")
print(f"Detected emotions: {detected_emotions}")
print(f"Overall Sentiment: {overall_sentiment}")

# Load Haar Cascade for face detection and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to calculate enhanced geometric features
def geometric_feature_extraction(face_rect):
    x, y, w, h = face_rect
    face_width = w
    face_height = h
    jaw_width = w * 0.75  # Approximation for jaw width (usually 75% of face width)
    
    # Ratio of face width to height
    face_ratio = face_width / face_height
    
    # Additional features (example: jaw width, overall face shape)
    return face_width, face_height, jaw_width, face_ratio

# Function to detect hair features (long/short/bald)
def detect_hair(image, face_rect):
    x, y, w, h = face_rect
    # Crop the region around the head above the face to detect hair
    head_region = image[max(0, y - int(h * 0.5)):y, x:x + w]
    
    # Convert to grayscale and check for texture in the hair region
    gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
    
    # Use a Gaussian blur to smooth the image and reduce noise
    blurred_head = cv2.GaussianBlur(gray_head, (5, 5), 0)
    
    # Threshold the image to find darker areas (possible hair)
    _, hair_mask = cv2.threshold(blurred_head, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate the percentage of hair pixels in the head region
    hair_pixels = np.sum(hair_mask == 255)
    total_pixels = hair_mask.size
    
    hair_density = hair_pixels / total_pixels
    
    # Refined threshold for detecting long hair
    if hair_density > 0.25:
        return "Long Hair"
    else:
        return "Short Hair or Bald"

# Function to check for facial hair (beard, mustache)
def detect_facial_hair(image, face_rect):
    x, y, w, h = face_rect
    mouth_region = image[y + int(0.6 * h):y + h, x:x + w]
    
    # Convert to grayscale and check for texture (facial hair detection)
    gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    
    # Apply a Gaussian blur to smooth out the region
    blurred_mouth = cv2.GaussianBlur(gray_mouth, (5, 5), 0)
    
    # Threshold the region to identify darker textures (facial hair)
    _, mouth_mask = cv2.threshold(blurred_mouth, 75, 255, cv2.THRESH_BINARY_INV)
    
    # Count pixels indicating facial hair (texture around the mouth)
    facial_hair_pixels = np.sum(mouth_mask == 255)
    
    # Refined threshold for detecting facial hair
    if facial_hair_pixels > 800:  # Lower the threshold slightly
        return "Facial Hair"
    return "No Facial Hair"

# Function to classify gender using geometric and image features
def classify_gender(face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair):
    # Combine geometric and feature-based classification
    if hair_feature == "Long Hair" and facial_hair == "No Facial Hair":
        return "Female"
    elif hair_feature == "Short Hair or Bald" and facial_hair == "Facial Hair":
        return "Male"
    else:
        # Fallback to geometric features if ambiguous
        if face_ratio > 0.85 and jaw_width > 60:
            return "Male"
        else:
            return "Female"

# List to store gender detection results for each image
results = []

# Directory containing the images for gender detection
image_dir = 'genderdetect_img.jpg'
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]

# Process only the first 1000 images
#image_files = image_files[:1000]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue

    # Convert image to grayscale for face detection and feature extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image using Haar Cascade
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Geometric feature extraction
        face_width, face_height, jaw_width, face_ratio = geometric_feature_extraction((x, y, w, h))

        # Detect hair features (long/short/bald)
        hair_feature = detect_hair(image, (x, y, w, h))
        
        # Detect facial hair (beard/mustache)
        facial_hair = detect_facial_hair(image, (x, y, w, h))

        # Classify gender based on all available features
        detected_gender = classify_gender(face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair)

        # Annotate the image with the detected gender
        cv2.putText(image, detected_gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the result image in the output folder
        output_path = os.path.join(output_folder, f"{image_file.split('.')[0]}_gender_detected.jpg")
        cv2.imwrite(output_path, image)

        # Save the results to list
        results.append([image_file, detected_gender, face_width, face_height, jaw_width, face_ratio, hair_feature, facial_hair])

# Save the detection results to a CSV file (optional)
csv_path = os.path.join(output_folder, "gender_identification_results.csv")
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Name', 'Detected Gender', 'Face Width', 'Face Height', 'Jaw Width', 'Face Ratio', 'Hair Feature', 'Facial Hair'])
    writer.writerows(results)

print("Enhanced Gender Identification Complete! Results saved in:", output_folder)





