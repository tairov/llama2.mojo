## llama2.🔥

<p align="center">
  <img src="assets/llama2.mojo-demo.gif" width="700" alt="llama2.mojo logo">
</p>

## why this port?

This repository serves as a port that provides a Mojo-based implementation of `llama2.c`.

With the release of [Mojo](https://www.modular.com/blog/mojo-its-finally-here), I was inspired to take my Python port of [llama2.py](https://github.com/tairov/llama2.py) and transition it to Mojo. The result? A version that leverages Mojo's SIMD & vectorization primitives, boosting the Python performance by nearly 250x. Impressively, the Mojo version now outperforms the original llama2.c, even in runfast mode, by 15-20%. This showcases the potential of hardware-level optimizations through Mojo's advanced features. I think this also can help us to see how far can we go with the original llama2.c hardware optimizations.

## performance

As it was shown during my experimentations performance of this solution can beat the original `llama2.c` even built
with `runfast` option
Ubuntu virtual machine performance:

| Model           | llama2.py | llama2.c    | llama2.c (runfast) | **llama2.mojo** | llama2.mojo (naive matmul) |
|-----------------|-----------|-------------|--------------------|-----------------|----------------------------|
| stories15M.bin  | 1.3 tok/s | 75.73 tok/s | 237 tok/s          | 260 tok/s       | 67.26 tok/s                | 
| stories110M.bin | -         | 9 tok/s     | 30 tok/s           | 40 tok/s        | 9.20                       | 

## prerequisites

Make sure you have installed & configured mojo on your environment.
https://docs.modular.com/mojo/manual/get-started/index.html

Or you can use [mojo playground](https://playground.modular.com/) to run this model.

## feel the 🔥 magic

First, navigate to the folder when you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/tairov/llama2.mojo.git
```

Then, open the repository folder:

```bash
cd llama2.mojo
```

Now, let's download the model

```bash
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

Then, just run the Mojo

```bash
mojo llama2.mojo
num hardware threads:  6  SIMD vector width:  8
checkpoint size:  60816028
<s>
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
Lily found a stick and tried to hit the ball. But the stick was too short. She tried again and again, but she couldn't reach it. She felt sad.
Suddenly, a kind man came by and saw Lily. He asked her what was wrong. Lily told him about the ball. The man smiled and said, "I have a useful idea!" He took out a long stick and used it to knock the ball down. Lily was so happy! She thanked the man and they played together in the sunshine.
<s>
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red
achieved tok/s:  264.24870466321244
```

## License

MIT