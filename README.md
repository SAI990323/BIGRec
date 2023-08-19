# Grounding4Rec

### Preprocess
Please follow the process.ipynb in each data directory.

### Training on Single Domain
```
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=$1 python finetune_gen.py \
                    --base_model /home/sds/baokq/LLAMA/hugging_face_LLAMA_weights_7B/ \
                    --train_data_path "[\"./data/movie/train.json\"]"   \
                    --val_data_path "[\"./data/movie/valid_5000.json"]" \
                    --output_dir /model/movie/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
```
### Training on Multi Domain
```
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=$1 python finetune_gen.py \
                    --base_model /home/sds/baokq/LLAMA/hugging_face_LLAMA_weights_7B/ \
--train_data_path "[\"./data/movie/train.json\", \"./data/game/train.json\"]"  \
                    --val_data_path "[\"./data/movie/valid_5000.json\", \"./data/game/valid_5000.json\"]"  \
                    --output_dir ./model/multi/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
                    
```

### Training on Multiple GPU Card
We provide our accelerate config in ./config/accelerate.yaml
```
accelerate config # Please set up your config
for seed in 0 1 2
do
    for lr in 1e-4
    do
        for dropout in 0.05    
        do
            for sample in 1024
            do
                echo "lr: $lr, dropout: $dropout , seed: $seed,"
                CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_gen.py \
                    --base_model /home/sds/baokq/LLAMA/hugging_face_LLAMA_weights_7B/ \
--train_data_path "[\"./data/movie/train.json\", \"./data/game/train.json\"]"  \
                    --val_data_path "[\"./data/movie/valid_5000.json\", \"./data/game/valid_5000.json\"]"  \
                    --output_dir ./model/multi/${seed}_${sample} \
                    --batch_size 128 \
                    --micro_batch_size 4 \
                    --num_epochs 50 \
                    --learning_rate $lr \
                    --cutoff_len 512 \
                    --lora_r 8 \
                    --lora_alpha 16\
                    --lora_dropout $dropout \
                    --lora_target_modules '[q_proj,v_proj]' \
                    --train_on_inputs \
                    --group_by_length \
                    --resume_from_checkpoint 'XXX' \
                    --seed $seed \
                    --sample $sample 
            done    
        done
    done
done
```