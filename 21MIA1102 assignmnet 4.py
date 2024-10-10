
import cv2 as cv
import matplotlib.pyplot as plt

def grayscale(frame):
    grey_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    return grey_frame

def spatiotemporalsegmentation(grey_frame):
    threshold1 = 100
    threshold2 = 200
    edges = cv.Canny(grey_frame, threshold1, threshold2)
    return edges

def scene_cut_detection(frame, previous_histogram, scene_cut_threshold=0.2):
    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    current_histogram = cv.calcHist([hsv_frame], [0], None, [256], [0, 256])

    if previous_histogram is not None:
        hist_diff = cv.compareHist(previous_histogram, current_histogram, cv.HISTCMP_CORREL)
        if hist_diff < scene_cut_threshold:
            return current_histogram, True  
    return current_histogram, False  

def visualize_results(frame, edges, is_scene_cut, frame_number):
    """Display original frame and edge detection results only for scene cuts."""
    plt.figure(figsize=(15, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    plt.title(f'Original Frame {frame_number}')
    plt.axis('off') 

   

    # Display edge detection result
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray') 
    plt.title(f'Edge Detection Frame {frame_number}')
    plt.axis('off')  

    # Highlight scene cut detection
    if is_scene_cut:
        plt.suptitle('Scene Cut Detected!', fontsize=16, color='red')

    plt.tight_layout()
    plt.show()

def main(file_path):
    video = cv.VideoCapture(file_path)
    if not video.isOpened():
        print(f"Error: Unable to open video {file_path}")
        return
    
    frame_number = 0 
    previous_histogram = None  
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Convert to grayscale and perform edge detection
        grey_frame = grayscale(frame)
        edges = spatiotemporalsegmentation(grey_frame)

        # Scene cut detection
        previous_histogram, is_scene_cut = scene_cut_detection(frame, previous_histogram)

        # Display only if scene cut is detected or for every 50th frame
        if is_scene_cut or frame_number % 50 == 0:
            visualize_results(frame, edges, is_scene_cut, frame_number)

        frame_number += 1 
    
    video.release()
    cv.destroyAllWindows()

main('videofile.mp4')
