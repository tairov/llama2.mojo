## llama2.🔥

<p align="center">
  <img src="assets/llama2.mojo-demo.gif" width="700" alt="llama2.mojo logo">
</p>

## why this port?

This repository serves as a port that provides a Mojo-based implementation of `llama2.c`.

With the release of [Mojo](https://www.modular.com/blog/mojo-its-finally-here), I was inspired to take my Python port
of [llama2.py](https://github.com/tairov/llama2.py) and transition it to Mojo. The result? A version that leverages
Mojo's SIMD & vectorization primitives, boosting the Python performance by nearly 250x. Impressively, the Mojo version
now outperforms the original `llama2.c` compiled in `runfast` mode out of the box by 15-20%.
This showcases the potential of hardware-level optimizations through Mojo's advanced features.
I think this also can help us to see how far can we go with the original `llama2.c` hardware optimizations.

## performance

Since there were some debates was this comparison legit or not I did some research and found that in `runfast`
mode `llama2.c`
includes multiple optimizations like aggressive vectorization, which makes comparison fair with Mojo SIMD vectorization.

Further researches of both solutions in parallelized mode compilation showed that `llama2.c` is faster by ~20%
I'm still investigating in this direction since not all the possible optimizations were applied to the Mojo version so
far.

### comparison

#### OS/HW specs

```
OS:         Ubuntu 20.04
CPU(s):     6
Model name: Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
CPU MHz:    3191.998
```

| Model           | [llama2.py](https://github.com/tairov/llama2.py) | [llama2.c](https://github.com/karpathy/llama2.c) | [llama2.c](https://github.com/karpathy/llama2.c) (runfast) | [llama2.c](https://github.com/karpathy/llama2.c) (OMP/parallelized) | **llama2.mojo** | **llama2.mojo** (parallelized) | llama2.mojo (naive matmul) |
|-----------------|--------------------------------------------------|--------------------------------------------------|------------------------------------------------------------|---------------------------------------------------------------------|-----------------|--------------------------------|----------------------------|
| stories15M.bin  | 1.3 tok/s                                        | 75.73 tok/s                                      | 237 tok/s                                                  | 450 tok/s                                                           | 260 tok/s       | 390 tok/s                      | 67.26 tok/s                | 
| stories110M.bin | -                                                | 9 tok/s                                          | 30 tok/s                                                   | 64 tok/s                                                            | 40 tok/s        | 57 tok/s                       | 9.20 tok/s                 | 

## prerequisites

Make sure you have installed and [configured mojo on your environment](https://docs.modular.com/mojo/manual/get-started/index.html)

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
mojo llama2.mojo stories15M.bin -s 100 -n 256 -t 0.5 -i "Llama is an animal"
```

**Example output**

```
num hardware threads:  6
SIMD vector width:  16
checkpoint size:  60816028
Llama is an animal was walking down the street. She stopped and looked up with a big smile on her face. She had a puppy in her arms. She was so excited to have a new friend.
The puppy ran up to her and said, "Hi! I'm here to be your friend!"
Mandy smiled and said, "Hi! I'm Mandy. Can I play with you?"
The puppy barked and wagged his tail. Mandy was so happy! She gave the puppy a big hug and they played with the puppy all afternoon.
When it was time to go home, Mandy said, "I have to go now. Goodbye!"
The puppy barked and said, "Goodbye Mandy! See you tomorrow!"
Mandy waved goodbye and then she went back home. She was so happy to have a new friend.
<s>
Once upon a time, there was a little girl named Lily. She loved to play outside and explore the world around her. One day, she went for a walk in the park with her mommy. They saw a big tree with lots of leaves.
Lily said,
achieved tok/s:  359.66149506346966
```

## Running via Docker

```bash
docker build -t llama2.mojo .
docker run -it llama2.mojo
```

With Gradio UI:

```bash
# uncomment the last line in Dockerfile CMD ["python", "gradio_app.py"]
docker run -it -p 0.0.0.0:7860:7860 llama2.mojo
``` 

## License

MIT