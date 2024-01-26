HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python ./vit/run_image_classification_quant.py \
    --output_dir ./quant_eval_outputs/ \
    --remove_unused_columns False \
    --do_eval \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --train_dir  "/nvme/wangh/data/ImageNet/train" \
    --validation_dir "/nvme/wangh/data/ImageNet/val" \
    
    # --train_dir  "/nvme/wangh/data/beans/train" \
    # --validation_dir "/nvme/wangh/data/beans/validation" \
    
    
    
    