from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import math
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
            path.append(os.path.join(args.input_dir, name))
print(path)

base_model = "YOUR_LLAMA_PATH"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

f = open('./id2name.txt', 'r')
items = f.readlines()
item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
item_ids = [_ for _ in range(len(item_names))]
item_dict = dict()
for i in range(len(item_names)):
    if item_names[i] not in item_dict:
        item_dict[item_names[i]] = [item_ids[i]]
    else:   
        item_dict[item_names[i]].append(item_ids[i])
    
topk_list = [1, 3, 5, 10, 20, 50]
result_dict = dict()
def get_dist(test_data):
    f.close()
    text = [_["predict"][0].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    for i, batch_input in tqdm(enumerate(batch(text, 8))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    movie_embedding = torch.load("./item_embedding.pt").cuda()
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)
    dist_min = torch.min(dist, dim=1, keepdim=True)[0]
    dist_max = torch.max(dist, dim=1, keepdim=True)[0]
    dist_norm = (dist - dist_min) / (dist_max - dist_min)
    return dist_norm

def get_ndcg(test_data, rank):
    NDCG = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_movie = test_data[i]['output'].strip("\"").strip(" ")
            rankId = 200000
            for _ in item_dict[target_movie]:
                rankId = min(rankId, rank[i][_].item())
            if rankId < topk:
                S = S + (1 / math.log(rankId + 2))
        NDCG.append(S / len(test_data) / (1 / math.log(2)))
    return NDCG

def get_hr(test_data, rank):
    HR = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_movie = test_data[i]['output'].strip("\"").strip(" ")
            rankId = 200000
            for _ in item_dict[target_movie]:
                rankId = min(rankId, rank[i][_].item())
            if rankId < topk:
                S = S + 1
        HR.append(S / len(test_data))
    return HR

for p in path:
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    key = p[:-5].split('_')[-2]

    f = open(p, 'r')
    import json
    test_data = json.load(f)
    dist = get_dist(test_data)


    ci_rank = torch.load(f'YOUR_SASRec_MODEL_DIR/games/SASRec/{key}/val.pt')
    assert ci_rank.shape == dist.shape
    ci_min = torch.min(ci_rank, dim=1, keepdim=True)[0]
    ci_max = torch.max(ci_rank, dim=1, keepdim=True)[0]
    ci_norm = (ci_rank - ci_min) / (ci_max - ci_min)
    ci_rank = ci_norm
    max_ndcg = [0 for _ in range(6)]
    max_hr = [0 for _ in range(6)]
    max_gamma_ndcg = [-1 for _ in range(6)]
    max_gamma_hr = [-1 for _ in range(6)]
    from tqdm import tqdm

    g_list = [(i + 1) / 100 for i in range(100)] + [i for i in range(1, 100)]
    for g in tqdm(g_list):
        # gamma = g / 100
        gamma = g
        rank = torch.pow((1 + ci_rank), -gamma) * dist
    
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        NDCG = get_ndcg(test_data, rank)
        HR = get_hr(test_data, rank)

        for i in range(6):
            if NDCG[i] > max_ndcg[i]:
                max_ndcg[i] = NDCG[i]
                max_gamma_ndcg[i] = gamma
            if HR[i] > max_hr[i]:
                max_hr[i] = HR[i]
                max_gamma_hr[i] = gamma
        
    print(max_ndcg)
    print(max_gamma_ndcg)
    print(max_hr)
    print(max_gamma_hr)


    f = open(p.replace('valid_', ''), 'r')
    import json
    test_data = json.load(f)
    dist = get_dist(test_data)


    ci_rank = torch.load(f'YOUR_SASRec_MODEL_DIR/games/SASRec/{key}/test.pt'))
    assert ci_rank.shape == dist.shape
    ci_min = torch.min(ci_rank, dim=1, keepdim=True)[0]
    ci_max = torch.max(ci_rank, dim=1, keepdim=True)[0]
    ci_norm = (ci_rank - ci_min) / (ci_max - ci_min)
    ci_rank = ci_norm
    
    
    Best_NDCG = []
    Best_HR = []
    for i in range(6):
        gamma = max_gamma_ndcg[i]
        rank = torch.pow((1 + ci_rank), -gamma) * dist
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        NDCG = get_ndcg(test_data, rank)
        Best_NDCG.append(NDCG[i])
        gamma = max_gamma_hr[i]
        rank = torch.pow((1 + ci_rank), -gamma) * dist
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        HR = get_hr(test_data, rank)
        Best_HR.append(HR[i])
    result_dict[p]["NDCG"] = Best_NDCG
    result_dict[p]["HR"] = Best_HR
    print(Best_NDCG)
    print(Best_HR)

f = open('./game_ci.json', 'w')    
json.dump(result_dict, f, indent=4)
f.close()

