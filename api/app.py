from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
interpreter = tf.lite.Interpreter(model_path=os.path.join(BASE_DIR, "models", "behavior_model.tflite"))
interpreter.allocate_tensors()

MEAN = [200.5806, 0.4057, 450.4667, 179.2505, 0.0493]
STD  = [29.3626,  0.0798, 78.6365,  41.0643,  0.0192]

EWMA_ALPHA = 0.15
BASELINE_DIR = os.path.join(BASE_DIR, "baselines")
os.makedirs(BASELINE_DIR, exist_ok=True)

FEATURE_KEYS = [
    'keystroke_avg_ms',
    'touch_pressure_avg',
    'swipe_avg_px_per_sec',
    'scroll_rhythm_ms',
    'accelerometer_avg_variance',
]


def get_baseline_path(user_id: str) -> str:
    return os.path.join(BASELINE_DIR, f"{user_id}.json")


def load_baseline(user_id: str) -> dict | None:
    path = get_baseline_path(user_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def save_baseline(user_id: str, baseline: dict):
    with open(get_baseline_path(user_id), "w") as f:
        json.dump(baseline, f)


def update_ewma_baseline(user_id: str, current_features: list[float]) -> dict:
    """
    Update (or initialize) the EWMA baseline for a user.
    newBaseline = (ALPHA × current) + (1 - ALPHA) × old
    Returns the updated baseline dict.
    """
    existing = load_baseline(user_id)

    if existing is None:
        # First session — initialise baseline directly
        new_values = current_features
        session_count = 1
    else:
        old_values = existing["values"]
        new_values = [
            EWMA_ALPHA * curr + (1 - EWMA_ALPHA) * old
            for curr, old in zip(current_features, old_values)
        ]
        session_count = existing.get("session_count", 1) + 1

    baseline = {"values": new_values, "session_count": session_count}
    save_baseline(user_id, baseline)
    return baseline


def score_against_baseline(raw: list[float], baseline_values: list[float]) -> float:
    """
    Compute a normalised deviation score between current features
    and the user's personal EWMA baseline (0 = identical, higher = more deviant).
    """
    deviations = [abs(raw[i] - baseline_values[i]) / STD[i] for i in range(len(raw))]
    return float(np.mean(deviations))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data.get('user_id', 'default')

    raw = [data[k] for k in FEATURE_KEYS]

    # ── Global-model inference ──────────────────────────────────────────────
    normalized = [(raw[i] - MEAN[i]) / STD[i] for i in range(5)]
    input_data = np.array([normalized], dtype=np.float32)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    global_score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

    # ── EWMA baseline scoring ───────────────────────────────────────────────
    baseline      = load_baseline(user_id)
    baseline_score = (
        score_against_baseline(raw, baseline["values"]) if baseline else None
    )

    # ── Combined anomaly decision ───────────────────────────────────────────
    # Blend global model + personal baseline once we have one
    if baseline_score is not None:
        combined_score = 0.6 * global_score + 0.4 * min(baseline_score / 3.0, 1.0)
    else:
        combined_score = global_score

    is_anomaly = combined_score > 0.8

    # ── Update baseline only on non-anomalous sessions ──────────────────────
    if not is_anomaly:
        updated_baseline = update_ewma_baseline(user_id, raw)
        session_count    = updated_baseline["session_count"]
    else:
        session_count = baseline["session_count"] if baseline else 0

    return jsonify({
        'score':          round(combined_score, 4),
        'global_score':   round(global_score, 4),
        'baseline_score': round(baseline_score, 4) if baseline_score is not None else None,
        'is_anomaly':     is_anomaly,
        'level':          'ANOMALY' if combined_score > 0.8 else 'ELEVATED' if combined_score > 0.5 else 'NORMAL',
        'session_count':  session_count,
        'baseline_updated': not is_anomaly,
    })


@app.route('/baseline/<user_id>', methods=['GET'])
def get_baseline(user_id):
    baseline = load_baseline(user_id)
    if baseline is None:
        return jsonify({'error': 'No baseline found'}), 404
    return jsonify({
        'user_id':       user_id,
        'session_count': baseline['session_count'],
        'baseline':      dict(zip(FEATURE_KEYS, baseline['values'])),
    })


@app.route('/baseline/<user_id>', methods=['DELETE'])
def reset_baseline(user_id):
    path = get_baseline_path(user_id)
    if os.path.exists(path):
        os.remove(path)
        return jsonify({'message': f'Baseline for {user_id} reset.'})
    return jsonify({'error': 'No baseline found'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)