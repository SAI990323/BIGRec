# Grounding4Rec

For item embedding, due to the quota of the git LFS, you can use the [link](https://rec.ustc.edu.cn/share/78de1e20-763a-11ee-b439-a3ef6ed8b1a3) with password 0g1g.
### Environment
```
pip install -r requirements.txt
```

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
                CUDA_VISIBLE_DEVICES=$1 python train.py \
                    --base_model YOUR_LLAMA_PATH/ \
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
                CUDA_VISIBLE_DEVICES=$1 python train.py \
                    --base_model YOUR_LLAMA_PATH/ \
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
                CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch train.py \
                    --base_model YOUR_LLAMA_PATH/ \
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

### Inference
```
#  Taking movie as an example
python inference.py \
    --base_model YOUR_LLAMA_PATH/ \
    --lora_weights YOUR_LORA_PATH \
    --test_data_path ./data/movie/test/test_5000.json \
    --result_json_data ./movie_result/movie.json
```

### Evaluate
```
# Taking Game as an example
# Directly
python ./data/movie/evaluate.py --input_dir ./movie_result
# CI Augmented
python ./data/movie/adjust_ci.py --input_dir ./movie_result # Note that you need to have your own SASRec/DROS model (Specify the path in the code)
```
