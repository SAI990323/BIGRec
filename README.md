# Grounding4Rec

　<font color=FF0000> We recently conducted tests on code performance on the platforms RTX3090, A40, and A100, and observed significant variations in code behavior across different environments. The reported results are based on running the code on a single A100. Currently, we are investigating the underlying factors contributing to these differences.  </font>

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
                CUDA_VISIBLE_DEVICES=$1 python finetune_gen.py \
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
                CUDA_VISIBLE_DEVICES=$1 python finetune_gen.py \
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
                CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch finetune_gen.py \
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
