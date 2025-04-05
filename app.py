from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Changed to static directory
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = load_model('models/shadow_removal_model.h5')

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = (img / 127.5) - 1
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    return (prediction + 1) * 127.5

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            # Save to static/uploads
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Process image
            processed_img = process_image(upload_path)
            output_filename = f'processed_{filename}'
            output_path = os.path.join('static/results', output_filename)
            cv2.imwrite(output_path, processed_img)
            
            # Use relative paths for template
            return render_template('index.html',
                                 original=filename,  # Just the filename
                                 processed=output_filename)

    return render_template('index.html')

# Route to serve uploaded files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('static/results', exist_ok=True)
    app.run(debug=False)
    port = int(os.environ.get('PORT', 10000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port)