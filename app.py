import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt

from recommendation.recommend_model import recommend_models


def analyze_dataset(file):
    result = recommend_models(file.name)

    models = result["recommended_models"]

    output_text = ""
    model_names = []
    accuracies = []
    times = []

    for m in models:
        output_text += f"""
### {m['rank']}. {m['model']}
Accuracy: {m['predicted_accuracy']}
Training Time: {m['predicted_training_time']} s  
Confidence: {m['confidence_score']}  

*{m['reason']}*

---
"""
        model_names.append(m["model"])
        accuracies.append(m["predicted_accuracy"])
        times.append(m["predicted_training_time"])

    # Create chart
    fig, ax1 = plt.subplots()

    ax1.bar(model_names, accuracies, label="Accuracy")
    ax1.set_ylabel("Accuracy")

    ax2 = ax1.twinx()
    ax2.bar(model_names, times, alpha=0.4, label="Training Time")
    ax2.set_ylabel("Training Time")

    plt.title("Accuracy vs Training Time Tradeoff")
    plt.xticks(rotation=20)

    return output_text, fig


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("#  Meta ML Recommender")

    with gr.Row():
        file_input = gr.File(label="Upload Dataset")
        analyze_btn = gr.Button("Analyze")

    output_markdown = gr.Markdown()
    output_chart = gr.Plot()

    analyze_btn.click(
        analyze_dataset,
        inputs=file_input,
        outputs=[output_markdown, output_chart]
    )

demo.launch()
