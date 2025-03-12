import cv2
import numpy as np
import argparse

# Set up command-line argument for the image path
parser = argparse.ArgumentParser(description='Detect symbols in P&ID image')
parser.add_argument('image_path', type=str, help='Path to the P&ID image')
args = parser.parse_args()

# Read and preprocess the image
img = cv2.imread(args.image_path)
if img is None:
    print(f"Error: Could not load image from {args.image_path}")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Edge detection for line analysis
edges = cv2.Canny(gray, 50, 150)

# Find contours using RETR_LIST to detect all contours, including nested ones
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Define the background color (assuming white; adjust if needed)
background_color = (255, 255, 255)  # BGR format: White (common in P&ID)

# Create an output image filled with the background color
output = np.full_like(img, background_color)

# Get image dimensions and set size constraints
img_height, img_width = img.shape[:2]
max_width = img_width / 2
max_height = img_height / 2

# Function to classify symbols with adaptive epsilon
def classify_symbol(contour, edges):
    x, y, w, h = cv2.boundingRect(contour)
    if w > max_width or h > max_height:
        return None
        
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    aspect = float(w) / h if h != 0 else 0
    roi = edges[y:y+h, x:x+w]
    
    # Adaptive epsilon: smaller for smaller perimeters
    epsilon = 0.01 * perimeter if perimeter > 100 else 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    
    # Circle detection
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    if vertices >= 6 and circularity > 0.7:
        if area < 200:
            return "Small Circle"
        return "Circle"
    
    # Triangle detection
    if vertices == 3:
        return "Triangle"
    
    # Other shapes (e.g., square, rectangle)
    if vertices == 4:
        if 0.95 < aspect < 1.05:
            return "Square"
        return "Rectangle"
    
    # Line types (optional for P&ID context)
    line_thickness = np.mean(roi) if roi.size > 0 else 0
    if line_thickness > 10:
        if cv2.countNonZero(roi) / (w * h) > 0.3:
            return "Thick Line"
        return "Solid Line"
    elif line_thickness > 2:
        if cv2.countNonZero(roi) / (w * h) < 0.2:
            return "Dashed Line"
        return "Thin Line"
    
    # P&ID-specific symbols (examples)
    if area > 1000:
        if vertices == 4 and 0.8 < aspect < 1.2:
            return "Tank/Vessel"
        elif vertices > 6 and 0.8 < aspect < 1.2:
            return "Instrument Circle"
    elif area > 200:
        if vertices == 3 and aspect > 1.5:
            return "Arrow"
        elif vertices == 4 and 0.8 < aspect < 1.2:
            return "Cross"
    
    return "Unknown"

# Detect and bound symbols
symbols = {}
min_area = 10  # Minimum area to detect small symbols (including triangles)

for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(contour)
        symbol_type = classify_symbol(contour, edges)
        
        if symbol_type and w <= max_width and h <= max_height:
            if symbol_type not in symbols:
                symbols[symbol_type] = []
            symbols[symbol_type].append((x, y, w, h))
            
            # Copy the symbol region from the original image to the output
            output[y:y+h, x:x+w] = img[y:y+h, x:x+w]

# Save the result
cv2.imwrite('pid_symbols_only.jpg', output)

# Print detected symbols summary
print("\nDetected Symbols (coordinates only):")
for symbol_type, boxes in symbols.items():
    print(f"{symbol_type}: {len(boxes)} instances")
    for i, (x, y, w, h) in enumerate(boxes, 1):
        print(f"  {i}. Position: ({x}, {y}), Size: ({w}x{h})")

# Display the result
cv2.imshow('P&ID Symbols Only', output)
cv2.waitKey(0)
cv2.destroyAllWindows()