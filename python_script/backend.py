import base64

from flask import Flask, jsonify, request
import os
from flask_cors import CORS
from pathlib import Path
from autotune import main

app = Flask(__name__)

CORS(app)
CORS(app, origins=['http://localhost:3000'])


@app.route('/api/process-audio', methods=['POST'])
def process_audio():
    # Check if request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent.'}), 400

    # Get the file from the request
    file = request.files['file']
    print(request.form.get('sliderValue'))
    print(request.form.get('selectedOption'))

    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # Check if the file is an audio file
    allowed_extensions = {'mp3', 'wav', 'ogg', 'flac'}
    if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension. Only mp3, wav, ogg, and flac are allowed.'}), 400

    # Save the file to a temporary directory
    filename = file.filename
    folder_path = './audios_raw'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, filename)
    file.save(file_path)

    # Process audio
    main(file_path, smoothing=1)
    filepath = Path(file_path)
    file_path = filepath.parent / (filepath.stem + '_pitch_corrected' + filepath.suffix)

    # Return the processed audio file as base64-encoded data
    with open(file_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({'audioData': data})


if __name__ == '__main__':
    app.run(debug=True)
