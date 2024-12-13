import os,sys

sys.path.append(os.getcwd())
os.environ['HF_HOME'] = '/mnt/swordfish-pool2/ccu/mukur-video-cache/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/swordfish-pool2/ccu/mukur-video-cache/hf_cache'

import torch
from transformers import AutoTokenizer, GemmaForCausalLM, GemmaConfig
import lm_eval
from lm_eval.loggers import WandbLogger
import argparse
import json
import wandb


def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)
    return total_size/(1024 ** 2)



def main(args):

    run = wandb.init(
        project="hpml_MQAdapat", 
        config = {
            "mqa-layer-config": args.mqa_layer_config,
            "ckpt": args.model_ckpt,
            "batch_size": args.batch_size
        },
        name = args.variant,

    )

    
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained=/mnt/swordfish-pool2/mukur/test_experiment/MQAdapt/{args.model_ckpt}",
        tasks=args.tasks,
        log_samples=False,
        # limit=0.02,
        device = args.device,
        batch_size = args.batch_size,
    )

    results['results']['model_size'] = get_directory_size(
        f"/mnt/swordfish-pool2/mukur/test_experiment/MQAdapt/{args.model_ckpt}"
        )

    print(results['results'])
    
    with open(os.path.join(args.output_dir, args.variant), 'w') as f:
        json.dump(results['results'], f)

    run.log(results['results'])
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MQAdap Checkpoint')
    parser.add_argument('--variant', type=str, default = "sens")
    parser.add_argument('--model-ckpt', type=str)
    parser.add_argument('--tasks', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--dtype', type=str, default = "float16")
    parser.add_argument('--mqa-layer-config', type=str)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    args.device = f"cuda:{args.gpu_id}"

    # if args.variant is None:
    bool_flags_mqa = [int(x) for x in args.mqa_layer_config.split(",")]
    if args.variant == 'sens':
        args.variant = f"{args.variant}_{[idx for idx,j in enumerate(bool_flags_mqa) if j == 1][0]}"
    elif args.variant == 'greedy_bs_end':
        args.variant = f"{args.variant}_{len([idx for idx,j in enumerate(bool_flags_mqa) if j == 1])}"
    elif args.variant == 'greedy_alt_end':
        args.variant = f"{args.variant}_{len([idx for idx,j in enumerate(bool_flags_mqa) if j == 1])}"


    main(args)

