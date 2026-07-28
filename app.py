import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

# ==========================================================
# Load Scaler and TensorFlow Model
# ==========================================================
try:
    scaler = joblib.load('breast_cancer_scaler.pkl')
    model = tf.keras.models.load_model('breast_cancer_model.h5')
    print('Scaler and model loaded successfully!')
except Exception as e:
    print(f'Error loading files: {e}')
    scaler = None
    model = None

# ==========================================================
# Prediction Function
# ==========================================================
def predict_cancer(radius, texture, perimeter, area,
                   smoothness, compactness, concavity,
                   concave_points, symmetry, fractal_dimension):

    if scaler is None or model is None:
        return '❌ Model or scaler failed to load.'

    try:
        # 10 user inputs (mean features)
        mean_features = [
            radius, texture, perimeter, area,
            smoothness, compactness, concavity,
            concave_points, symmetry, fractal_dimension
        ]

        # Pre-assumed error features
        error_features = [
            0.2204, 0.8561, 1.778, 16.64,
            0.00508, 0.01104, 0.0, 0.0,
            0.01344, 0.001784
        ]

        # Pre-assumed worst features
        worst_features = [0.0] * 10

        # Total 30 features
        features = mean_features + error_features + worst_features

        # Convert to numpy array
        input_array = np.array([features])

        print('Input shape:', input_array.shape)
        print('Scaler expects:', scaler.n_features_in_)

        # Scale input
        scaled_input = scaler.transform(input_array)

        # Predict
        probability = model.predict(scaled_input, verbose=0)[0][0]

        if probability >= 0.5:
            return (
                f'🟢 BENIGN\\n\\n'
                f'Confidence: {probability:.2%}\\n\\n'
                f'The tumour is likely non-cancerous.'
            )
        else:
            malignant_confidence = 1 - probability
            return (
                f'🔴 MALIGNANT\\n\\n'
                f'Confidence: {malignant_confidence:.2%}\\n\\n'
                f'The tumour may be cancerous. Please consult a doctor.'
            )

    except Exception:
        return traceback.format_exc()

# ==========================================================
# Gradio Interface
# ==========================================================
with gr.Blocks() as app:

    gr.Markdown('# 🔬 Breast Cancer Detection System')
    gr.Markdown('Adjust the medical values below and run the analysis.')

    with gr.Row():
        with gr.Column():
            radius = gr.Slider(0, 40, value=14.0, step=0.1, label='Mean Radius')
            texture = gr.Slider(0, 50, value=19.0, step=0.1, label='Mean Texture')
            perimeter = gr.Slider(0, 200, value=90.0, step=1, label='Mean Perimeter')
            area = gr.Slider(0, 3000, value=650.0, step=10, label='Mean Area')
            smoothness = gr.Slider(0.0, 0.2, value=0.09, step=0.001, label='Mean Smoothness')

        with gr.Column():
            compactness = gr.Slider(0.0, 0.5, value=0.10, step=0.001, label='Mean Compactness')
            concavity = gr.Slider(0.0, 0.5, value=0.08, step=0.001, label='Mean Concavity')
            concave_points = gr.Slider(0.0, 0.25, value=0.04, step=0.001, label='Mean Concave Points')
            symmetry = gr.Slider(0.0, 0.5, value=0.18, step=0.001, label='Mean Symmetry')
            fractal_dimension = gr.Slider(0.0, 0.15, value=0.06, step=0.001, label='Mean Fractal Dimension')

    result = gr.Textbox(label='Prediction Result', lines=5)

    inputs = [
        radius, texture, perimeter, area,
        smoothness, compactness, concavity,
        concave_points, symmetry, fractal_dimension
    ]

    with gr.Row():
        predict_btn = gr.Button('Run Analysis', variant='primary')
        clear_btn = gr.ClearButton(components=inputs + [result])

    predict_btn.click(
        fn=predict_cancer,
        inputs=inputs,
        outputs=result
    )

# ==========================================================
# Launch for Render
# ==========================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print(f'Starting on port {port}...')

    app.launch(
        server_name='0.0.0.0',
        server_port=port,
        show_error=True,
        quiet=False
    )
