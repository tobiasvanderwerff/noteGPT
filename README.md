## Problem statement

I often write time-stamped notes for various things I want to document. A typical note
would look as follows:

```
# dd-mm-yyyy

## Meeting with John

I talked to John today and he told me about an interesting idea he had. It goes as
follows: ...
```

As the size of my notes grows, I end up not coming back to these notes, because it's
hard to know where to look for specific ideas. So in the end this information goes to
waste. Instead, I would like to use an LLM assistant to point me to relevant notes when
I ask about a specific subject. For example: 

```
User: Tell me about stereo image matching ideas

GPT: Here are some relevant search results.

01-01-2022 
File: /home/user/notes/random/notes.md
======
In this note, you mention that a sliding window might be an idea to find pixel
correspondences.


09-05-2022 
File: /home/user/notes/other/notes.md
======
It is mentioned that sliding window is not good enough, and that a more sophisticated
method is needed. Using neural networks for depth estimation might be something to
consider.
```

## Requirements

### Model requirements
There are two models required: an **LLM**, and an **embedding model** (for text search).
Overall, no internet connection should be required to run the entire program.

#### LLM
- The LLM model should run locally, due to privacy concerns. Using the OpenAI API is not
  ideal if your notes may contain sensitive information (who knows what they do with
  your data!).
- The LLM should run on at most 6GB of GPU memory (which is what I'm working with), or
  run on CPU.

#### Embedding model
- Should run locally

### Response requirements

The requirements for the response are enforced by using prompt engineering. Basically,
try to feed the LLM an explanation of its task, in a way which maximally optimizes for
task performance. The response requirements are:

- The LLM should mention a specific note, along with its date and filename.
- The LLM should summarize briefly why the note is relevant for the given query.


## Models

The only way in which I've been able to use sufficiently large models (7B+), is by using
8-bit or 4-bit quantization, using [LLama.cpp](https://github.com/ggerganov/llama.cpp).
Models I've tried:

### LLMs
- **Llama-7b**: not very usable because it's not instruction-tuned, i.e., it's a base
  model
- **Vicuna-7b**: producing some nice results so far
- **Falcon-7b-instruct**: TBD
- **MPT-7b**: TBD

### Embedding models

## Workflow

There's a lot of nice tools out there for working with LLMs (e.g. LangChain).
The main thing that is needed is a suitable LLM that is good enough as an
assistant, which, roughly speaking, means it should be able to accurately answer
questions about unstructured pieces of text. My workflow so far:

1. Download an interesting model off of Huggingface (consider using [this script](https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py) to download it)
1. Use `llama.cpp` to quantize it to 4-bit representation (see [the repo](https://github.com/ggerganov/llama.cpp)) for instructions.
1. Plug the model into [text generation webui](https://github.com/oobabooga/text-generation-webui) for some quick experimentation before we introduce more complexity, e.g. using embeddings for text search
1. ...

## Data generation

To generate some sample data to try out, I used GPT4 to generate a bunch of notes (Bing
Chat in creative mode). The prompt I used is in `gpt4_prompt.txt`.


## Inspiration for this project
- https://github.com/imartinez/privateGPT
- https://github.com/PromtEngineer/localGPT
  - basically a fork of https://github.com/imartinez/privateGPT
  - uses Vicuna-7B
  - without quanization, Vicuna-7b is unusable because it uses way too much memory (not
    even GPU memory, but plain old RAM). With 32 GB of RAM, the model cannot even be
    loaded properly 


## Ideas
- give some context to the LLM. E.g. when I say "YOLO", I want it to know it has to do
  with object detection.
- save query results in some way to make the model smarter over time 


## Resources
- For prompt inspiration, the "Building Systems with the ChatGPT API" course:
  https://learn.deeplearning.ai/chatgpt-building-system/
