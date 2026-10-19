import os
import joblib
import numpy as np
import pandas as pd
import gradio as gr

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "traffic_flow_prediction_tuned_random_forest.joblib"
)

def load_artifacts():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Cannot find {MODEL_PATH}. Put the trained .joblib model file "
            "in the same folder as app.py."
        )
    saved = joblib.load(MODEL_PATH)
    if not isinstance(saved, dict):
        raise ValueError(
            "Expected a dictionary model package containing model and feature_columns."
        )
    model = saved.get("model")
    columns = saved.get("feature_columns") or saved.get("selected_features")
    if model is None or not columns:
        raise ValueError(
            "The saved package must include 'model' and 'feature_columns' "
            "(or 'selected_features')."
        )
    return saved, model, list(columns)

try:
    PACKAGE, MODEL, FEATURES = load_artifacts()
    LOAD_ERROR = None
except Exception as exc:
    PACKAGE, MODEL, FEATURES = None, None, []
    LOAD_ERROR = str(exc)

def predict(*values):
    if MODEL is None:
        raise gr.Error("Model not loaded: " + str(LOAD_ERROR))
    row = pd.DataFrame([dict(zip(FEATURES, values))], columns=FEATURES)
    data = row

    imputer = PACKAGE.get("imputer")
    if imputer is not None:
        data = imputer.transform(data)

    selector = PACKAGE.get("feature_selector_model")
    if selector is not None:
        data = selector.transform(data)

    result = np.asarray(MODEL.predict(data)).reshape(-1)[0]
    return f"Predicted traffic flow: {float(result):.4f}"

if MODEL is None:
    with gr.Blocks(title="Traffic Flow Prediction") as demo:
        gr.Markdown("# Traffic Flow Prediction Dashboard")
        gr.Markdown(
            "The app file is ready, but the trained model could not be loaded. "
            "Upload the expected .joblib model file beside app.py."
        )
        gr.Markdown(f"**Details:** {LOAD_ERROR}")
else:
    with gr.Blocks(title="Traffic Flow Prediction") as demo:
        gr.Markdown("# Traffic Flow Prediction Dashboard")
        gr.Markdown("Enter a value for each feature, then click **Predict**.")
        inputs = [gr.Number(label=str(feature), value=0) for feature in FEATURES]
        output = gr.Textbox(label="Prediction", interactive=False)
        button = gr.Button("Predict", variant="primary")
        button.click(fn=predict, inputs=inputs, outputs=output)
        gr.Markdown(f"Model: {PACKAGE.get('model_name', 'Saved traffic-flow model')}")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        share=False
    )
