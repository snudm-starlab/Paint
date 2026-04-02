gpu=$1

python main_quant.py \
    --gpu "$gpu$" \
    --wandb \
    --project_name "Paint (Data-free quantization)" \
    --dataset <path_to_imagenet_dataset> \
    --datapool <path_to_save_synthesized_images> \
    --model deit_tiny_16_imagenet \
    --iteration 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --cutoff_ratio 0.7 \
    --lambda_anchor 1.0 \
    --soft_label_alpha 0.9
