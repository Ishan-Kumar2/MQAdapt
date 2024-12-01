import torch
from transformers import AutoTokenizer, GemmaForCausalLM, GemmaConfig
import argparse

def mqadapt_weights(model_weights, bool_flags_mqa, num_layers, num_attention_heads):
    updated_weights = {}
    for layer in range(num_layers):
        do_mqa = bool_flags_mqa[layer]
        
        if do_mqa == 1:
            for key in ["k_proj", "v_proj"]:
                weight_key = f"model.layers.{layer}.self_attn.{key}.weight"
                if weight_key in model_weights:
                    original_weight = model_weights[weight_key]
                    
                    # The projection matrix weights are stored as [num_attention_heads*head_dim, input_dim]
                    head_dim = original_weight.shape[0] // num_attention_heads
                    
                    original_weight = original_weight.view(num_attention_heads, head_dim, -1)
                    # Now the weights are [num_attention_heads, head_dim, input_dim]
                    
                    original_weight = original_weight.mean(dim=0)
                    # Now the weights are [1, head_dim, input_dim]
                    
                    #Get rid of this line
                    # original_weight = original_weight.repeat(num_attention_heads, 1, 1).reshape(num_attention_heads * head_dim, -1)
                    updated_weights[weight_key] = original_weight
    ## Copy the rest as is
    for k, v in model_weights.items():
        if k not in updated_weights:
            updated_weights[k] = v
    return updated_weights
    
def modify_and_save_gemma(model_name, bool_flags_mqa, save_path):
    
    print(f"Loading model: {model_name}")
    model = GemmaForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.float16)

    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads

    print("Adapting weights for multi-query attention...")
    updated_weights = mqadapt_weights(model.state_dict(), bool_flags_mqa, num_layers, num_attention_heads)
    
    
    configuration = GemmaConfig(mqa_layers = bool_flags_mqa)
    print("Updating model weights...")
    new_model = GemmaForCausalLM(configuration)
    new_model.load_state_dict(updated_weights)

    print(f"Saving the modified model to: {save_path}")
    new_model.save_pretrained(save_path)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify Gemma model weights for MQA.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the Gemma model to load.")
    parser.add_argument("--bool_flags_mqa", type=str, required=True, help="Binary list (as a string) for MQA flags.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the modified model.")
    args = parser.parse_args()
    bool_flags_mqa = [int(x) for x in args.bool_flags_mqa.split(",")]
    print(bool_flags_mqa)
    modify_and_save_gemma(args.model_name, bool_flags_mqa, args.save_path)
    
    
# python script.py --model_name google/gemma-7b --num_layers 32 --bool_flags_mqa 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0 --save_path ./newGemma.pt
