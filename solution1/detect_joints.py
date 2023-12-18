import cv2
import numpy as np

def detect_joints_export(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Pre-processing: Contrast enhancement
    img_eq = cv2.equalizeHist(image)
    
    # Edge detection: Canny operator
    edges = cv2.Canny(img_eq, 50, 150)
    
    # Binarization
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    
    # Morphological operations: Closing to connect edges
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Feature extraction: Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create an image to draw the joints
    joints_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw contours and find intersections as potential joint locations
    for cnt in contours:
        # Approximate contour to simplify it
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Only consider contours with more than one point after approximation
        if len(approx) > 1:
            for point in approx:
                x, y = point[0]
                cv2.circle(joints_img, (x, y), 5, (0, 0, 255), -1)

    # Export the image with detected joints
    export_path = 'detected_joints_exported.png'
    cv2.imwrite(export_path, joints_img)

    return export_path

# Run the function and get the export path
export_path = detect_joints_export('untitled1.png')


