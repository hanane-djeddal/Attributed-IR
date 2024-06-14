# An Evaluation Framework for Attributed Information Retrieval using Large Language Models

Search engines are now adopting generative approaches to provide  answers along with in-line citations as attribution. While existing work focuses mainly on attributed question answering (QA), we target information-seeking scenarios which are often more challenging. This repository maintains a framework to evaluate and benchmark attributed information seeking. We implement three key architectures commonly used for attribution, extendable to any LLM or LLM-based approach: (1) Generate (2) Retrieve then Generate, and (3) Generate then Retrieve. We also provide several automatic evaluation metrics for attributed information seeking from literature and propose possible adptations. Baselines and experiments are run using [HAGRID](https://github.com/project-miracl/hagrid) dataset two LLMs : [Zephyr 3b](https://huggingface.co/stabilityai/stablelm-zephyr-3b) and [LLaMA-2-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) . But can easily extend to other LLMs and other datasets for the task.

![](media/attribution_architectures.png)

## Content
1. [Installation](#installation)
2. [Architectures](#architectures)
3. [Evaluation](#evaluation)
4. [Custom Data](#customdata)
5. [Results](#results)
6. [Contact](#contact)
## Installation
Install dependent Python libraries by running the command below.

```
pip install -r requirements.txt
```

You can also create a conda environment by running the command below.

```
conda env create -f attributed-ir.yml
```
## Architectures

The different architectures of LLM can be found in `scripts`. The configurations are already stored in `config.py` but you can adjust as needed.
#### Generate (G)

A simple run of the script specifying the architecture and the model :

```
python generate_answer.py  --architcture G --model_name zephyr
```

#### Retrieve Then Generate (RTG)
For this architecture, we need first to retrieve the documents, then use them for answer generation
##### RTG-gold
If the dataset has annotated relevant documents i.e. gold documents,  we can run the generation directly without retrieval.
```
python generate_answer.py  --architcture RTG-gold --model_name zephyr 
```
##### RTG-vanilla
First run the retrieval script :

```
python retrieve.py
```

Once the retrieval is done, you can update `config.py` with the name of the generated answer . You can then run : 
```
python generate_answer.py --architcture RTG-vanilla --model_name zephyr
```

##### RTG-query-gen
First generate the queries:

```
python generate_queries.py  --model_name zephyr
```
Then we retrieve using the generated queries. By default the aggregation method is "rerank" but can be modified in `config.py`.

```
python retrieve_with_generated_queries.py
```

Then run the answer generation script specifying the architecture .
```
python generate_answer.py  --architcture RTG-query-gen --model_name zephyr
```

#### Generate Then Retrieve (GTR)
If you have not run the experiment G (Generate), you have to do so. Then run :

```
python retrieve_posthoc_gtr.py
```




## Evaluation

We evaluate both the correctness and attribution of the answer. We take in consideration the case where multiple answers are possible. We also offer the evaluation of retrieval results

To evaluate correctness run 

```
python evaluate_Correstness_answer_with_citation.py
```

For attribution metrics : 

```
python citations_eval.py
```

For retrieval:

```
python evaluate_retrieval.py
```

## Custom Data

We use HAGRID dataset to run our experiments, but this code can be easily applicable to other datasets from huggingfce or from custom files, provided that the dataset contains these fields: 

query : the question or the query

answers : list of possible gold answers (can be one or more)

quotes : documents used as context to generate the answer (Not needed for architecture Generate (G))

You can specify the name of your data file in `config.py`> `data_path` and change what each column is called in the dataset in `config.py`> `column_names`


## Results

*Gold answer* reports the citation quality in the gold answer using automatic metrics. Since the scenario *G* does not include citations, only answer correctness is evaluated. Since the generated answer in scenario *GTR* is the same as in scenario *G*, the answer correctness measures are equal. The best performance measures are in **bold**.
<sub><sup>
<table >
  <tr>
    <th rowspan="2">   Scenarios   </th> 
    <th colspan="7">Correctness</th>
    <th colspan="6">Citations</th>
  </tr>
  <tr>
    <th>BLEU</th>
    <th>ROUGE Prec</th>
    <th>ROUGE Rec</th>
    <th>ROUGE F-score</th>
    <th>BertScore Prec</th>
    <th>BertScore Rec</th>
    <th>BertScore F-score</th>
    <th>Overlap Prec</th>
    <th>Overlap Rec</th>
    <th>AutoAIS Cit.</th>
    <th>AutoAIS Pssg.</th>
    <th>NLI Prec.</th>
    <th>NLI Rec.</th>
  </tr>
  <tr>
    <td>Gold answer</td>
    <td></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>87.97</td>
    <td>89.21</td>
    <td>83.65</td>
    <td>79.80</td>
  </tr>
  <tr>
    <td>G</td>
    <td>11.06</td>
    <td>30.41</td>
    <td>46.08</td>
    <td>31.58</td>
    <td>87.88</td>
    <td>90.02</td>
    <td>88.87</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>RTG - gold</td>
    <td><b>28.22</b></td>
    <td><b>44.00</b></td>
    <td><b>63.81</b></td>
    <td><b>46.72</b></td>
    <td><b>90.02</b></td>
    <td><b>93.36</b></td>
    <td><b>91.69</b></td>
    <td><b>75.29</b></td>
    <td><b>68.89</b></td>
    <td><b>42.81</b></td>
    <td><b>80.67</b></td>
    <td>56.55</td>
    <td><i>42.31</i></td>
  </tr>
  <tr>
    <td>RTG - vanilla (2-psg)</td>
    <td><i>18.44</i></td>
    <td><i>33.83</i></td>
    <td><i>56.40</i></td>
    <td><i>36.65</i></td>
    <td><i>87.94</i></td>
    <td>91.52</td>
    <td><i>89.63</i></td>
    <td><i>36.17</i></td>
    <td><i>32.69</i></td>
    <td>41.86</td>
    <td>78.95</td>
    <td><i>57.90</i></td>
    <td>41.63</td>
  </tr>
  <tr>
    <td>RTG - query-gen (2-psg)</td>
    <td>18.33</td>
    <td>33.58</td>
    <td>56.13</td>
    <td>36.43</td>
    <td>87.89</td>
    <td><i>91.55</i></td>
    <td>89.62</td>
    <td>35.89</td>
    <td>32.46</td>
    <td><i>42.68</i></td>
    <td><i>80.10</i></td>
    <td><b>59.59</b></td>
    <td><b>42.48</b></td>
  </tr>
  <tr>
    <td>GTR (1-psg)</td>
    <td>11.06</td>
    <td>30.41</td>
    <td>46.08</td>
    <td>31.58</td>
    <td>87.88</td>
    <td>90.02</td>
    <td>88.87</td>
    <td>45.53</td>
    <td>30.53</td>
    <td>26.69</td>
    <td>26
    </td>
  </tr>
</table>
</sub></sup>


## Contact
If you have questions, please open an issue mentioning @hanane-djeddal or send an email to hanane.djeddal[at]irit.fr.

