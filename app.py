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
    # We must load BOTH the scaler and the neural network
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
def predict_cancer(*features):
    # Features are passed in as a tuple of 30 items. We convert to a list.
    values = list(features)

    # 1. Empty input check
    if any(v is None or str(v).strip() == "" for v in values):
        return "❌ Please fill in all 30 medical measurements."

    # 2. Type casting to float
    try:
        float_values = [float(v) for v in values]
    except (ValueError, TypeError) as e:
        return f"❌ Data Conversion Error. All inputs must be numbers.\n\nDetails: {str(e)}"

    # 3. Model execution
    if deployed_nn is None or scaler is None:
        return "❌ Server Error: Model or Scaler failed to load. Check your repository files."

    try:
        # Convert the single row of 30 features into a 2D NumPy array
        input_array = np.array([float_values])

        # --- CODE BLOCK: APPLY SCALING BEFORE PREDICTION ---
        # The Neural Network was trained on scaled data, so we MUST scale the user's raw input
        scaled_input = scaler.transform(input_array)
        # ---------------------------------------------------

        # Get the prediction probability from the Sigmoid activation function
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
# Interface Setup (Enhanced Tabbed Layout)
# ==========================================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate")) as app:
    
    gr.Markdown("<h1 style='text-align: center;'>🔬 Breast Cancer Detection System (Deep Learning)</h1>")
    gr.Markdown("<p style='text-align: center;'>Predict tumor classifications using a 30-feature Neural Network.</p>")
    gr.Markdown("---")

    # Layout: Using Tabs to organize 30 inputs cleanly
    with gr.Tabs():
        
        # TAB 1: Mean Measurements
        with gr.TabItem("1. Mean Metrics"):
            with gr.Row():
                with gr.Column():
                    f1 = gr.Number(label="Mean Radius")
                    f2 = gr.Number(label="Mean Texture")
                    f3 = gr.Number(label="Mean Perimeter")
                    f4 = gr.Number(label="Mean Area")
                    f5 = gr.Number(label="Mean Smoothness")
                with gr.Column():
                    f6 = gr.Number(label="Mean Compactness")
                    f7 = gr.Number(label="Mean Concavity")
                    f8 = gr.Number(label="Mean Concave Points")
                    f9 = gr.Number(label="Mean Symmetry")
                    f10 = gr.Number(label="Mean Fractal Dimension")

        # TAB 2: Error Measurements
        with gr.TabItem("2. Error Metrics"):
            with gr.Row():
                with gr.Column():
                    f11 = gr.Number(label="Radius Error")
                    f12 = gr.Number(label="Texture Error")
                    f13 = gr.Number(label="Perimeter Error")
                    f14 = gr.Number(label="Area Error")
                    f15 = gr.Number(label="Smoothness Error")
                with gr.Column():
                    f16 = gr.Number(label="Compactness Error")
                    f17 = gr.Number(label="Concavity Error")
                    f18 = gr.Number(label="Concave Points Error")
                    f19 = gr.Number(label="Symmetry Error")
                    f20 = gr.Number(label="Fractal Dimension Error")

        # TAB 3: Worst Measurements
        with gr.TabItem("3. Worst Metrics"):
            with gr.Row():
                with gr.Column():
                    f21 = gr.Number(label="Worst Radius")
                    f22 = gr.Number(label="Worst Texture")
                    f23 = gr.Number(label="Worst Perimeter")
                    f24 = gr.Number(label="Worst Area")
                    f25 = gr.Number(label="Worst Smoothness")
                with gr.Column():
                    f26 = gr.Number(label="Worst Compactness")
                    f27 = gr.Number(label="Worst Concavity")
                    f28 = gr.Number(label="Worst Concave Points")
                    f29 = gr.Number(label="Worst Symmetry")
                    f30 = gr.Number(label="Worst Fractal Dimension")

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

    # Wire up the logic exactly in the order of the dataset
    input_components = [
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
        f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
    ]
    
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
