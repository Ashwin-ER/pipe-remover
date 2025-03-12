import cv2
import numpy as np
from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def paint_lines_white(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=5)
    
    # Create a mask to store the lines
    mask = np.zeros_like(image)
    
    # Draw the detected lines in white
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=2)
    
    # Identify non-symbol areas by filtering out the detected lines
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
    
    # Invert the mask for the symbols
    symbols_mask = cv2.bitwise_not(mask_binary)
    
    # Preserve symbols by combining with the original image
    modified_image = image.copy()
    modified_image[np.where(mask_binary == 255)] = [255, 255, 255]
    
    # Save the output
    cv2.imwrite(output_path, modified_image)
    return output_path

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No file selected', 400
    
    if file:
        # Secure the filename and create file paths
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_' + filename)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_' + filename)
        
        # Save the uploaded file
        file.save(input_path)
        
        # Process the image
        try:
            paint_lines_white(input_path, output_path)
            
            # Clean up input file
            os.remove(input_path)
            
            # Return the processed image
            return send_file(output_path, as_attachment=True, download_name='processed_' + filename)
        except Exception as e:
            return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)