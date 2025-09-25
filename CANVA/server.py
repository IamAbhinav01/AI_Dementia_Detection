from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from pathlib import Path

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # allow all origins

@app.route('/save_capture', methods=['POST'])
def save_capture():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        # Decode the base64 image data (remove "data:image/png;base64," prefix)
        image_str = data['image']
        if "," in image_str:
            image_str = image_str.split(",")[1]

        image_data = base64.b64decode(image_str)

        # Create captures directory if it doesn't exist
        captures_dir = Path(__file__).parent / "captures"
        captures_dir.mkdir(exist_ok=True)

        # Save the image
        image_path = captures_dir / "capture.png"
        with open(image_path, 'wb') as f:
            f.write(image_data)

        print(f"Image saved at {image_path}")
        return jsonify({'message': 'Image saved successfully as capture.png'}), 200

    except Exception as e:
        print("Error saving image:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
