import sys, os, load_dotenv
load_dotenv.load_dotenv()
try:
    from my_gpt.app.chat import chatapp_interface
    from my_gpt.app.benchmark import benchmark_interface
except ImportError as e:
    print("Import Error:", e)
    if os.environ["ENV"] == "development":
        print("Development environment detected. Attempting to adjust sys.path.")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(current_dir))
        sys.path.append(parent_dir)
        print("Updated System Path:", sys.path)
        from my_gpt.app.chat import chatapp_interface
        from my_gpt.app.benchmark import benchmark_interface
    else:
        raise e
import gradio as gr
from pydantic import BaseModel


class ChatSettings(BaseModel):
    temperature: float = 1.0
    max_tokens: int = 64
    model_name: str = ""

class ModelSettings(BaseModel):
    nb_parameters_min: int = "1B"
    nb_parameters_max: int = "175B"


with gr.Blocks(title="My-GPT") as app:
    with gr.Tab("Chat"):
        chatapp_interface()

    with gr.Tab("Benchmark"):
        benchmark_interface()

    with gr.Tab("Training"):
        gr.Markdown("# Training Interface 🏋️‍♂️")


if __name__ == "__main__":
    app.launch()