from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "models", "behavior_model.tflite"))
interpreter.allocate_tensors()

MEAN = [200.5806, 0.4057, 450.4667, 179.2505, 0.0493]
STD =  [29.3626,  0.0798, 78.6365,  41.0643,  0.0192]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw = [
        data['keystroke_avg_ms'],
        data['touch_pressure_avg'],
        data['swipe_avg_px_per_sec'],
        data['scroll_rhythm_ms'],
        data['accelerometer_avg_variance'],
    ]
    normalized = [(raw[i] - MEAN[i]) / STD[i] for i in range(5)]
    input_data = np.array([normalized], dtype=np.float32)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
    
    return jsonify({
        'score': score,
        'is_anomaly': score > 0.8,
        'level': 'ANOMALY' if score > 0.8 else 'ELEVATED' if score > 0.5 else 'NORMAL'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)