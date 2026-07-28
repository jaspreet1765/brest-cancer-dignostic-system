# app.py

import os
import joblib
import traceback
import numpy as np
import gradio as gr
import tensorflow as tf

# ==========================================================
# Load the Scaler and TensorFlow Model
# ==========================================================
try:
    scaler = joblib.load('breast_cancer_scaler.pkl')
    deployed_nn = tf.keras.models.load_model('breast_cancer_model.h5')
    print("Scaler and Deep Learning Model loaded successfully!")
except Exception as e:
    print(f"Warning: Files not found or error loading. {e}")
    scaler = None
    deployed_nn = None

# ==========================================================
# Prediction Function with Bulletproof Error Handling
# ==========================================================
def predict_cancer(f1, f2, f3, f4, f5, f6, f7, f8, f9, f10):
    
    # 1. Capture the 10 user-provided Mean features from Sliders
    user_mean_features = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

    # 2. Hardcoded Error features
    preassumed_error_features = [
        0.2204,    # radius error mode
        0.8561,    # texture error mode
        1.778,     # perimeter error mode
        16.64,     # area error mode
        0.005080,  # smoothness error mode
        0.01104,   # compactness error mode
        0.0,       # concavity error mode
        0.0,       # concave points error mode
        0.01344,   # symmetry error mode
        0.001784   # fractal dimension error mode
    ]

    # --- CODE BLOCK: UPDATED WORST FEATURES (CHANGES MADE HERE) ---
    # 3. Hardcoded Worst features using the actual dataset modes you provided
    preassumed_worst_features = [
        12.36,    # worst radius mode
        17.70,    # worst texture mode 
        101.7,    # worst perimeter mode 
        284.4,    # worst area mode 
        0.1216,   # worst smoothness mode
        0.1486,   # worst compactness mode
        0.0,      # worst concavity mode
        0.0,      # worst concave points mode
        0.2226,   # worst symmetry mode
        0.07427   # worst fractal dimension mode
    ]
    # --------------------------------------------------------------

    # 4. Combine all arrays to perfectly match the 30 features the neural network expects
    full_30_features = user_mean_features + preassumed_error_features + preassumed_worst_features

    # Model execution
    if deployed_nn is None or scaler is None:
        return "❌ Server Error: Model or Scaler failed to load. Check your repository files."

    try:
        # Convert the full row of 30 features into a 2D NumPy array
        input_array = np.array([full_30_features])

        # Apply scaling before prediction
        scaled_input = scaler.transform(input_array)

        # Get the prediction probability
        prediction_prob = deployed_nn.predict(scaled_input)[0][0]

        # Scikit-learn Breast Cancer target mapping: 0 = Malignant, 1 = Benign
        if prediction_prob >= 0.5:
            return (
                f"🟢 Assessment Result (Confidence: {prediction_prob:.2%})\n\n"
                "Classification: BENIGN\n\n"
                "The cell characteristics suggest a non-cancerous tumor."
            )
        else:
            malignant_confidence = 1 - prediction_prob
            return (
                f"🔴 Assessment Result (Confidence: {malignant_confidence:.2%})\n\n"
                "Classification: MALIGNANT\n\n"
                "The cell characteristics indicate a high risk of cancer. Please consult an oncologist immediately."
            )

    except Exception as e:
        error_trace = traceback.format_exc()
        print("RUNTIME ERROR:\n", error_trace)
        return f"❌ Prediction failed due to an internal error.\n\nDEBUG INFO:\n{error_trace}"

# ==========================================================
# Interface Setup (Enhanced Slider Layout)
# ==========================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as app:
    
    gr.Markdown("<h1 style='text-align: center;'>🔬 Breast Cancer Detection System</h1>")
    gr.Markdown("<p style='text-align: center;'>Adjust the basic medical metrics below. Advanced metrics are automatically calculated.</p>")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            f1 = gr.Slider(minimum=0, maximum=40, step=0.1, value=14.0, label="Mean Radius")
            f2 = gr.Slider(minimum=0, maximum=50, step=0.1, value=19.0, label="Mean Texture")
            f3 = gr.Slider(minimum=0, maximum=200, step=1.0, value=90.0, label="Mean Perimeter")
            f4 = gr.Slider(minimum=0, maximum=3000, step=10.0, value=650.0, label="Mean Area")
            f5 = gr.Slider(minimum=0.0, maximum=0.2, step=0.001, value=0.09, label="Mean Smoothness")
        
        with gr.Column():
            f6 = gr.Slider(minimum=0.0, maximum=0.5, step=0.001, value=0.1, label="Mean Compactness")
            f7 = gr.Slider(minimum=0.0, maximum=0.5, step=0.001, value=0.08, label="Mean Concavity")
            f8 = gr.Slider(minimum=0.0, maximum=0.25, step=0.001, value=0.04, label="Mean Concave Points")
            f9 = gr.Slider(minimum=0.0, maximum=0.5, step=0.001, value=0.18, label="Mean Symmetry")
            f10 = gr.Slider(minimum=0.0, maximum=0.15, step=0.001, value=0.06, label="Mean Fractal Dimension")

    # Output Section
    gr.Markdown("---")
    with gr.Row():
        submit_btn = gr.Button("Run Neural Network Analysis", variant="primary", size="lg")
        clear_btn = gr.ClearButton(size="lg")
    
    with gr.Row():
        result_box = gr.Textbox(label="Deep Learning Assessment Result", lines=5, interactive=False)

    # Footer
    gr.Markdown("""
    ---
    ### 👨‍💻 About the Developer
    **Created by:** Jaspreet kaur (MERN Stack Developer & SDE)
    * **LinkedIn:** [Connect with me](YOUR_LINKEDIN_URL_HERE)
    * **GitHub:** [Check out my projects](YOUR_GITHUB_URL_HERE)
    """)

    # Wire up the logic mapped only to the 10 visible sliders
    input_components = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]
    
    submit_btn.click(fn=predict_cancer, inputs=input_components, outputs=result_box)
    clear_btn.add(input_components + [result_box])

# ==========================================================
# Launch Configuration
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting Gradio server on 0.0.0.0:{port}...")
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
    )
