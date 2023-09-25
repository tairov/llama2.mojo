import gradio as gr
import subprocess
import sys
from pathlib import Path

async def generate(prompt, model_name, seed=0, temperature=0.5, num_tokens=256):
    # stream stout
    base = ""#"../model/"
    tokenizer_name = "tokenizer.bin"
    if model_name == "tl-chat.bin":
        tokenizer_name = 'tok_tl-chat.bin'
    process = subprocess.Popen(
        [
            "mojo",
            "llama2.mojo",
            Path(base + model_name),
            "-s",
            str(seed),
            "-n",
            str(num_tokens),
            "-t",
            str(temperature),
            "-i",
            prompt,
            "-z",
            Path(base + tokenizer_name)
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    text = ""
    for char in iter(lambda: process.stdout.read(1), b""):
        char_decoded = char.decode("utf-8", errors="ignore")
        text += char_decoded
        yield text


with gr.Blocks() as demo:
    gr.Markdown(
        """
# llama2.🔥
## [Mojo](https://docs.modular.com/mojo/) implementation of [llama2.c](https://github.com/karpathy/llama2.c) by [@tairov](https://github.com/tairov)
Source: https://github.com/tairov/llama2.mojo
    """
    )
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Add your prompt here...")
            seed = gr.Slider(
                minimum=0,
                maximum=2**53,
                value=0,
                step=1,
                label="Seed",
                randomize=True,
            )
            temperature = gr.Slider(
                minimum=0.0, maximum=2.0, step=0.01, value=0.0, label="Temperature"
            )
            num_tokens = gr.Slider(
                minimum=1, maximum=256, value=256, label="Number of tokens"
            )
            model_name = gr.Dropdown(
                ["stories15M.bin", "stories42M.bin", "stories110M.bin", "tl-chat.bin"],
                value="stories15M.bin",
                label="Model Size",
            )
            with gr.Row():
                stop = gr.Button("Stop")
                run = gr.Button("Run")
        with gr.Column(scale=2):
            output_text = gr.Textbox(label="Generated Text")

    # update maximum number of tokens based on model size
    model_name.change(
        lambda x: gr.update(maximum=1024)
        if x == "stories110M.bin" or x == "stories42M.bin" or x == "tl-chat.bin"
        else gr.update(maximum=256),
        model_name,
        num_tokens,
        queue=False,
    )
    click_event = run.click(
        fn=generate,
        inputs=[prompt, model_name, seed, temperature, num_tokens],
        outputs=output_text,
    )
    stop.click(fn=None, inputs=None, outputs=None, cancels=[click_event])

demo.queue()
demo.launch(server_name="0.0.0.0")
