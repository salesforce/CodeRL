<p align="center">
  <img src="images/logo.jpg" width="50%">
</p>

## CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning <a name="corl"></a>


This is the official code for the paper **CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**.  

Authors:
Hung Le, Yue Wang, Akhilesh Deepak Gotmare, Silvio Savarese, Steven C.H. Hoi 


<!---
If you find the paper or the source code useful to your projects, please cite the following bibtex: 
<pre>
TBA
</pre>
-->

### Contents:
* [x] [CodeRL Overview](##coderl)
	* [x] [Abstract](###abstract)
	* [x] [Model Architecture](###model)
* [x] [Installation](##install)
* [x] [Datasets](##datasets)
	* [] [Example Unit Tests](###exampletests)
* [] [Models](##models)
* [] [Processes](##processes)  
	* [x] [Generating Programs](###generate)
	* [x] [Running Unit Tests](###runtests)
	* [x] [Evaluating Programs](###evaluate)
	* [] [Generating Programs with Critic Sampling](###criticsampling)
	* [] [Training CodeT5](###training)
* [x] [Example Generated Programs](##exampleprogram)
* [] [Citation] (##cite)
* [x] [License](#license) 

 
### Abstract <a name="abstract"></a>

Program synthesis or code generation aims to generate a program that satisfies a problem specification. Recent approaches using large-scale pretrained language models (LMs) have shown promising results, yet they have some critical limitations. In particular, they often follow a standard supervised fine-tuning procedure to train a
code generation model from natural language problem descriptions and ground-truth programs only. Such a paradigm largely ignores some important but potentially useful signals in the problem specification such as unit tests, 
which thus results in poor performance when solving complex unseen coding tasks.  To address the limitations, we propose **CodeRL**, a new framework for program synthesis tasks through pretrained LMs and deep reinforcement learning (RL). Specifically, during training, we treat the code-generating LM as an actor network, and introduce a critic network that is trained to predict the functional correctness of generated programs and provide dense feedback signals to the actor. During inference, we introduce a new generation procedure with a critical sampling strategy that allows a model to automatically regenerate programs based on feedback from example unit tests and critic scores. For the model backbones, we extended the encoder-decoder architecture of CodeT5 with enhanced learning objectives, larger model sizes, and better pretraining data. Our method not only achieves new SOTA results on the challenging APPS benchmark, but also shows strong zero-shot transfer capability with new SOTA results on the simpler MBPP benchmark. 

<p align="center">
<img src="images/coderl_overview.png" width="100%" />
 <br>
<b>An example program synthesis task (Right)</b>: Each task includes a problem specification in natural language, which often contains example input and output pairs. The expected output is a program that is checked for functional correctness against some unit tests. 
<b>A high-level overview of our CodeRL framework for program synthesis (Left)</b>: Our CodeRL framework treats pretrained language model (LM) as a stochastic policy, token predictions as actions, and rewards can be estimated based on unit test results of output programs
</p>


### Model Architecture <a name="model"></a>

<p align="center">
<img src="images/coderl_training.png" width="100%" />
<b>Overview of our actor-critic framework to optimize pretrained LMs for program
synthesis</b>: We treat the LM as an actor network and sample synthetic samples from this actor. Another neural network is trained as a critic model to evaluate these synthetic samples based on their probabilities of passing unit tests. The returns are estimated based on critic scores and finally factored into the RL objective to finetune the actor LM network using synthetic samples.
</p>

<p align="center">
<img src="images/coderl_inference.png" width="100%" />
<b>Overview of our Critic Sampling (CS) approach for program synthesis during inference</b>:
programs are refined and repaired based on their results on example unit tests of the corresponding problems. Program candidates are sampled by their critic-predicted scores at the token or sequence level. Dotted lines indicate optional processes that apply during program refining or repairing.
</p>


## Installation  <a name="install"></a>

The code requires some dependencies as specified in `requirements.txt`. Please follow the relevant libraries to install or run: 

`pip install -r requirements.txt`

## Datasets ## <a name="datasets"></a>

For pretraining, we use the [Python Github Code Dataset (GCPY)](https://huggingface.co/datasets/lvwerra/github-code). 
We filter the dataset by keeping only the code with licenses that at least permit academic use (“mit”, “apache-2”, “bsd-3-clause”, “bsd-2- 126 clause”, “cc0-1.0”, “unlicense”, “isc”). Please see the paper for more details on pretraining data preprocessing and pretraining. 

After pretraining, we finetune/evaluate models on the following major program synthesis benchmarks: 

* **APPS**: Please follow the downloading and preprocessing instructions provided [here](https://github.com/hendrycks/apps). 
* **MBPP**: The dataset is available [here](https://github.com/google-research/google-research/tree/master/mbpp). 

On both benchmarks, we follow the same way of preprocessing data and constructing input/output sequences as the original benchmark papers. 

Download and unzip all files into the `data` folder.

### Example Unit Tests <a name="exampletests"></a>
In addition to the original hidden unit tests on APPS, we also utilize the example tests that are often embedded in problem descriptions.
We will release the data with example unit tests we extracted on the APPS test split. 

## Models <a name="models"></a>

We will release the following pretrained/finetuned model checkpoints: 

* CodeT5: a CodeT5-770M model which is pretrained with Next Token Prediction learning objective and GCPY dataset 
* CodeRL+CodeT5: the above pretrained CodeT5 model which is finetuned on APPS following our CodeRL training framework 
* Critic: a CodeT5 model which is initialized from a CodeT5-base and trained as a classifier to predict unit test outcomes. The critic is used to estimate returns and facilitate RL finetuning. 

Download all files into the `models` folder.

## Processes <a name="processes"></a>

### Generating Programs <a name="generate"></a>

We created `scripts/generate.sh` to generate programs on the APPS benchmark. You can directly run this file by configuring the following parameters: 

|   **Parameters**  |                                              **Description**                                             |       **Example Values**       |
|:-----------------:|:--------------------------------------------------------------------------------------------------------:|:------------------------------:|
| `model_path`        | Path to a trained CodeT5-style model                                                                     | models/codet5\_finetuned_codeRL |
| `tokenizer_path`    | Path to the saved tokenizer for CodeT5 (or path to cache the tokenizer)                                  | models/codet5_tokenizer/       |
| `test_path`         | Path to the original test samples                                                                        | data/APPS/test/                |
| `start`             | start index of test samples to be generated                                                              | 0                              |
| `end`               | end index of test samples to be generated                                                                | 5000                           |
|`num_seqs`          | number of total output programs to be generated (for sampling generation)                                | 1000                           |
| `num_seqs_per_iter` | Depending on the limit of GPU, we can generate multiple rounds, each with this number of output programs | 50                             |
| `temp`              | temperature for sampling generation                                                                      | 0.6                            ||

Other parameters are defined in the file `utils/generate_config.py`.

Running the generation script will output programs, each of which is saved into a `json` file, including data fields `code` (list of output programs) and `prompt` (constructed input sequence to the LM model).


### Running Unit Tests  <a name="runtests"></a>

Once the programs are generated, they are evaluated against the corresponding unseen unit tests in each problem. 

To execute the unit tests and obtain test outcomes, we adapt our code to the official implementation of the [APPS benchmark](https://github.com/hendrycks/apps/tree/main/eval). 

We created `scripts/run_unit_tests.sh` to generate programs on the APPS benchmark. You can directly run this file by configuring the following parameters:

| **Parameters** |                                                                                **Description**                                                                               |                  **Example Values**                 |
|:--------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------:|
| `code_path`      | Path to the generated programs to be evaluated                                                                                                                               | outputs/codes/                                      |
| `output_path`    | Path to the save unit test results                                                                                                                                           | outputs/test_results/                               |
| `test_path`      | Path to the original test samples                                                                                                                                            | data/APPS/test/                                     |
| `example_tests`  | Whether to evaluate the programs on example unit tests (for filtering, refining programs) or hidden unit tests (for final evaluation)                                        | 0: use hidden unit tests; 1: use example unit tests |
| `start`          | start index of test samples to be evaluated                                                                                                                                  | 0                                                   |
| `end`            | end index of test samples to be evaluated                                                                                                                                    | 5000                                                |
| `threads`        | Depending on the capacity of the computation resource to run unit tests, we can run unit tests on multiple test samples over multiple threads to speed up the execution time | 30                                                  |


Running the script will output test results for each program. For each test sample, the results are saved into a `pickle` file, including data fields `results` (list of test outcomes, one of -2 = compile error, -1 = runtime error, False = failed test case, True = passed test case), `errors` (real compile error trace with details like error type and line numbers),  and `sols` (corresponding programs being evaluated).

Compared to the original implementation from APPS, we adopt one trick which will exit the unit testing loop if a program does not pass any test case. This will speed up the testing process while the final passing rate measures are not affected. Refer to the `run_test` function in `utils/testing_utils.py` for more details. 


### Evaluating Programs <a name="evaluate"></a>
To compute the pass@k metrics, rather than using the APPS evaluation metrics, we follow the official implementation of the [HumanEval benchmark](https://github.com/openai/human-eval) (which better measures pass@k normalized by the number of possible k programs)

### Generating Programs with Critic Sampling <a name="criticsampling"></a>

We will release the implementation details of our critic sampling procedure. 

### Training CodeT5 <a name="training"></a>

We will release the implementation details of model training/finetuning. 


## Example Generated Programs <a name="exampleprogram"></a>

<p align="center">
<img src="images/example_code.png" width="100%" />
The problem is from the APPS benchmark, and the solution programs are generated by CodeT5 and CodeRL.
</p>

## Citation <a name="cite"></a>



## License <a name="license"></a>

The code is released under BSD 3-Clause - see `LICENSE.txt` for details.

This code is developed from other open source projects: including [APPS](https://github.com/hendrycks/apps/), [HumanEval](https://github.com/openai/human-eval), and [transformers](https://github.com/huggingface/transformers). We thank the original contributors of these works for open-sourcing their valuable source codes. 

