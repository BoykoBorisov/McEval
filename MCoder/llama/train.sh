
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export MASTER_ADDR="localhost"
export MASTER_PORT="1231"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"



OUTPUT_PATH="/mnt/scratch/bborisov/models/coding/C#"


CUDA_VISIBLE_DEVICES="0,1"  
torchrun train.py \
    --lang "C#"\
    --bf16 True \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 40 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \


