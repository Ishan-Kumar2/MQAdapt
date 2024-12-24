# MQAdapt: Layerwise Adaptive Multi-Query Attention for efficient LLM inference

In this work we propose MQAdapt: an adaptive multi-query attention (MQA) approach to sample transformer layers to use inference-time MQA (Multi Query Attention) for computation speed-up with minimum performance drop.

We demonstrate results using [Gemma 7B (google/gemma-7b)](https://huggingface.co/google/gemma-7b) and [Llama 3B (meta-llama/Llama-3.2-3B)
](https://huggingface.co/meta-llama/Llama-3.2-3B)



| Model                               | Params      | Hellaswag average |
|-------------------------------------|-------------|--------------|
| Gemma 7B (google/gemma-7b)  | 7.0B  | 81.2         |
| Llama 3B (meta-llama/Llama-3.2-3B)          | 3B     | 67.0         |


## Getting Started
### 0. Set Environments
Before running the project, ensure you install the necessary dependencies. Use the following command to install all the required Python packages:
```bash
pip install -r requirements.txt
```

### 1. Conversion to MQA from MHA checkpoints
First, you need to convert the existing pretrained MHA checkpoint into you desired level of MQA checkpoint.
To define the layers which have MQA we use a list of bool flags with 0 meaning MHA and 1 meaning convert to MQA.
```
MQA_CONFIG="1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path "/path/to/save/model/"
```

### 2. Running evaluations
You can run a eval on your custom task, custom model using the eval script as follows.
```
python model_eval.py \
--model-ckpt "/path/to/save/" \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir "/path/to/save/eval_results/"

```

### 3. Run the evaluation studies from the paper
#### Layer Sensitivity Test
To judge the sensitivity of the layers to MQA, we do a search by applying MQA to one layer and moving that higher the transformer block.
```
./scripts/run_layer_sensitvity.sh
```
#### Greedy Search 
In an attempt to find the optimal number of layers we should convert to MQA, we started with a Binary search on number of layers to apply MQA to.
```
./scripts/greedy_binary_search_from_end.sh
```
#### Alternate Search
Same as the binary search experiment but with the key change that instead of consecutive layers we have the MQA on alternating layers.
```
./scripts/greedy_alt_from_end.sh
```

# Code Structure 
1. ```convertToMQAEfficient.py``` Contains the code which takes a MHA checkpoint and converts to layerwise MQA, where the layers are defined by the user. It saves the converted model as a huggingface model which can be loaded directly using the from_pretrained function of Huggingface transformers.
2. ```model_eval.py``` Contains the code to run eval and store results in Weights and Biases. It has the checkpoint directory, task and batch-size etc defined as params and can be configured using bash args.
3. ```/scripts``` Contains the scripts for performing the evaluation studies described in the paper : ```greedy_alt_from_end.sh```, ```greedy_binary_search_from_end.sh```, ```run_layer_sensitvity.sh```
4. ```/transformers/``` is the modified version of the HF transformers package with our change.

In case you want to try out MQAdapt with other Huggingface transformers, you need to replicate the changes to that transformer's modular code refer to [this commit](https://github.com/Ishan-Kumar2/MQAdapt/commit/09a4d1b90c74642076dc6ee91cb920ad53228a57) for Gemma. After that you need to create a new MQAdapt checkpoint by running Step 1 onwards in the getting started guide above.


# Results on Gemma 7B (HellaSwag)
## Sensitivity Analysis 
The lower transformer blocks have higher impact on the representation and hence when converted to MQA leader to bigger drop. 
The top layer has task specific info and therefore acts as an outlier to this trend. 

<img src="https://github.com/Ishan-Kumar2/MQAdapt/blob/main/sensitivity_evaluations/images/sensitivityGraph.png" width="700">

## Greedy Sampling
Layers dont respond well when we apply MQA to consecutive layers

<img src="https://github.com/Ishan-Kumar2/MQAdapt/blob/main/sensitivity_evaluations/images/GreedyGraph.png" width="700">

## Alternate Sampling
This indicates the layers are a lot more sensitive to MQA if they have MQA on previous layer as well. The effect (accuracy drop) gets normalised with a non-MQA layer in middle.

<img src="https://github.com/Ishan-Kumar2/MQAdapt/blob/main/sensitivity_evaluations/images/alternateGraph.png" width="700">
