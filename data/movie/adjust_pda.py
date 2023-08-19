from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import math
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
        if name.find("valid") > -1:
            path.append(os.path.join(args.input_dir, name))
print(path)

base_model = "YOUR_LLAMA_PATH"
tokenizer = LlamaTokenizer.from_pretrained(base_model)
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)

f = open('./movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1].strip("\"") for _ in movies]
movie_ids = [_ for _ in range(len(movie_names))]
movie_dict = dict(zip(movie_names, movie_ids))
origin_ids = [_.split('::')[0] for _ in movies]
id_mapping = dict(zip(origin_ids, range(len(origin_ids))))
    
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
    for i, batch_input in tqdm(enumerate(batch(text, 16))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids
        attention_mask = input.attention_mask
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    movie_embedding = torch.load("./movie_embedding.pt").cuda()
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
            if p.find("des") == -1:
                target_movie = test_data[i]['output'].strip("\"")
            else:
                target_movie = test_data[i]['movie_name'].strip("\"")
            target_movie_id = movie_dict[target_movie]
            rankId = rank[i][target_movie_id].item()
            if rankId < topk:
                S = S + (1 / math.log(rankId + 2))
        NDCG.append(S / len(test_data) / (1 / math.log(2)))
    return NDCG

def get_hr(test_data, rank):
    HR = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            if p.find("des") == -1:
                target_movie = test_data[i]['output'].strip("\"")
            else:
                target_movie = test_data[i]['movie_name'].strip("\"")
            target_movie_id = movie_dict[target_movie]
            rankId = rank[i][target_movie_id].item()
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
    f = open('./pop_count_real_time.json', 'r')
    pop_count = json.load(f)
    key = p[:-5].split('_')[-2] # seed
    pop_count = pop_count[key]
    pop_count = sorted(pop_count.items(), key=lambda x: x[1], reverse=True)
    pop_rank = torch.zeros([10681])
    for i in range(len(pop_count)):
        pop_rank[id_mapping[pop_count[i][0]]] = pop_count[i][1]
    pop_rank /= pop_rank.sum()
    pop_rank_origin = (pop_rank - pop_rank.min()) / (pop_rank.max() - pop_rank.min())
    print("popularity information-- mean:{},max:{},min:{}".format(pop_rank_origin.mean(), pop_rank_origin.max(), pop_rank_origin.min()))

    dist = get_dist(test_data)
    pop_rank = pop_rank_origin.cuda().repeat(dist.shape[0], 1).view(dist.shape[0], -1)
    max_ndcg = [0 for _ in range(6)]
    max_hr = [0 for _ in range(6)]
    max_gamma_ndcg = [-1 for _ in range(6)]
    max_gamma_hr = [-1 for _ in range(6)]
    for g in range(0, 10000, 100):
        gamma = g / 100
        rank = torch.pow((1 + pop_rank), -gamma) * dist
            
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        NDCG = get_ndcg(test_data, rank)
        HR = get_ndcg(test_data, rank)

        for i in range(6):
            if NDCG[i] > max_ndcg[i]:
                max_ndcg[i] = NDCG[i]
                max_gamma_ndcg[i] = gamma
            if HR[i] > max_hr[i]:
                max_hr[i] = HR[i]
                max_gamma_hr[i] = gamma
        
    
    print(max_gamma_ndcg)
    print(max_gamma_hr)

    f = open(p.replace('valid_',''), 'r')
    import json
    test_data = json.load(f)
    dist = get_dist(test_data)
    
    Best_NDCG = []
    Best_HR = []
    for i in range(6):
        gamma = max_gamma_ndcg[i]
        rank = torch.pow((1 + pop_rank), -gamma) * dist
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        NDCG = get_ndcg(test_data, rank)
        Best_NDCG.append(NDCG[i])
        gamma = max_gamma_hr[i]
        rank = torch.pow((1 + pop_rank), -gamma) * dist
        rank = rank.argsort(dim = -1).argsort(dim = -1)
        HR = get_hr(test_data, rank)
        Best_HR.append(HR[i])
    result_dict[p]["NDCG"] = Best_NDCG
    result_dict[p]["HR"] = Best_HR
    print(Best_HR)
    print(Best_NDCG)

f = open('./movie_pda.json', 'w')    
json.dump(result_dict, f, indent=4)
f.close()
