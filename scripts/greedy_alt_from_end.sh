
export CUDA_VISIBLE_DEVICES="5"


# rm -rf sensitivity_model_dir
# mkdir sensitivity_model_dir

# MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1"

# python convertToMQAEfficient.py --model_name google/gemma-7b \
# --bool_flags_mqa "${MQA_CONFIG}" \
# --save_path sensitivity_model_dir

# python model_eval.py --variant greedy_alt_end \
# --model-ckpt sensitivity_model_dir \
# --tasks hellaswag \
# --batch-size 16 \
# --gpu-id 0 \
# --mqa-layer-config "${MQA_CONFIG}" \
# --output-dir sensitivity_evaluations



# rm -rf sensitivity_model_dir
# mkdir sensitivity_model_dir

# MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1"

# python convertToMQAEfficient.py --model_name google/gemma-7b \
# --bool_flags_mqa "${MQA_CONFIG}" \
# --save_path sensitivity_model_dir

# python model_eval.py --variant greedy_alt_end \
# --model-ckpt sensitivity_model_dir \
# --tasks hellaswag \
# --batch-size 16 \
# --gpu-id 0 \
# --mqa-layer-config "${MQA_CONFIG}" \
# --output-dir sensitivity_evaluations



# rm -rf sensitivity_model_dir
# mkdir sensitivity_model_dir

# MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1"

# python convertToMQAEfficient.py --model_name google/gemma-7b \
# --bool_flags_mqa "${MQA_CONFIG}" \
# --save_path sensitivity_model_dir

# python model_eval.py --variant greedy_alt_end \
# --model-ckpt sensitivity_model_dir \
# --tasks hellaswag \
# --batch-size 16 \
# --gpu-id 0 \
# --mqa-layer-config "${MQA_CONFIG}" \
# --output-dir sensitivity_evaluations



rm -rf sensitivity_model_dir
mkdir sensitivity_model_dir

MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path sensitivity_model_dir

python model_eval.py --variant greedy_alt_end \
--model-ckpt sensitivity_model_dir \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir sensitivity_evaluations




rm -rf sensitivity_model_dir
mkdir sensitivity_model_dir

MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path sensitivity_model_dir

python model_eval.py --variant greedy_alt_end \
--model-ckpt sensitivity_model_dir \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir sensitivity_evaluations



rm -rf sensitivity_model_dir
mkdir sensitivity_model_dir

MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path sensitivity_model_dir

python model_eval.py --variant greedy_alt_end \
--model-ckpt sensitivity_model_dir \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir sensitivity_evaluations




rm -rf sensitivity_model_dir
mkdir sensitivity_model_dir

MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path sensitivity_model_dir

python model_eval.py --variant greedy_alt_end \
--model-ckpt sensitivity_model_dir \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir sensitivity_evaluations




rm -rf sensitivity_model_dir
mkdir sensitivity_model_dir

MQA_CONFIG="0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1"

python convertToMQAEfficient.py --model_name google/gemma-7b \
--bool_flags_mqa "${MQA_CONFIG}" \
--save_path sensitivity_model_dir

python model_eval.py --variant greedy_alt_end \
--model-ckpt sensitivity_model_dir \
--tasks hellaswag \
--batch-size 16 \
--gpu-id 0 \
--mqa-layer-config "${MQA_CONFIG}" \
--output-dir sensitivity_evaluations