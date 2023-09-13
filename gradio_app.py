import gradio as gr
import subprocess
import sys
import os


async def generate(prompt):
    # os.environ["PROMPT"] = prompt
    # stream stout
    process = subprocess.Popen(
        ["mojo", "llama2.mojo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    text = ""
    for char in iter(lambda: process.stdout.read(1), b""):
        char_decoded = char.decode()
        sys.stdout.write(char_decoded)
        text += char_decoded
        yield text


output_text = gr.Textbox(label="Generated Text")

demo = gr.Interface(
    fn=generate,
    inputs=None,
    outputs=output_text,
    description="""
# llama2.🔥
## [Mojo](https://docs.modular.com/mojo/) implementation of [llama2.c](https://github.com/karpathy/llama2.c) by [@tairov](https://github.com/tairov)
Source: https://github.com/tairov/llama2.mojo
    """,
    allow_flagging="never",
)

demo.queue()
demo.launch(server_name="0.0.0.0")
