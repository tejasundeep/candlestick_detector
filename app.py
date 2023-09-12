import gradio as gr
from ultralyticsplus import YOLO, render_result

# Metadata for the Model
title = "Candlestick Scanner"
img_component = gr.Image(type="pil")

# Function to Predict Trading Patterns
def predict_trading_pattern(input_image: gr.Image):
    model = YOLO('weights/best.pt')
    results = model.predict(input_image)
    return render_result(model=model, image=input_image, result=results[0])

# Initialize Gradio Interface
gradio_app = gr.Interface(
    fn=predict_trading_pattern,
    inputs=img_component,
    outputs=img_component,
    title=title,
    cache_examples=False,
)

gradio_app.launch()