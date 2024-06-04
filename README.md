# An Evaluation Framework for Attributed Information Retrieval using Large Language Models

Search engines are now adopting generative approaches to provide  answers along with in-line citations as attribution. While existing work focuses mainly on attributed question answering (QA), we target information-seeking scenarios which are often more challenging. This repository maintains a framework to evaluate and benchmark attributed information seeking. We implement three key architectures commonly used for attribution, extendable to any LLM or LLM-based approach: (1) Generate (2) Retrieve then Generate, and (3) Generate then Retrieve. We also provide several automatic evaluation metrics for attributed information seeking from literature and propose possible adptations. Baselines and experiments are run using [HAGRID](https://github.com/project-miracl/hagrid) dataset two LLMs : [Zephyr 3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) and [LLaMA-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) . But can easily extend to other LLMs and other datasets for the task.

![](media/attribution_architectures.png)

## Content
1. [Installation](#installation)
2. [Architectures](#architectures)
3. [Evaluation](#evaluation)
4. [Data](#data)
5. [Results](#results)
6. [FAQ](#faq)
7. [Contact](#contact)
## Installation
Install dependent Python libraries by running the command below.

```
pip install -r requirements.txt
```

You can also create a conda environment by running the command below.

```
conda env create -f attributed-ir.yml
```
