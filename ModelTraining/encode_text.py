import torch
import random
import numpy as np
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def interp_linear_beta(beta_alpha,n,save_path,emb_path):
    embed = torch.load(emb_path)
    beta_alpha = beta_alpha
    lambdas = np.random.beta(beta_alpha,beta_alpha,500000)
    mix_id=0
    for i in tqdm(embed.keys()):
        interpolation_examples = []
        for j in range(n):
            ij = random.sample(embed[i],2)
            mixed_tensor = ij[0][1]*lambdas[mix_id] + ij[1][1]*(1-lambdas[mix_id])
            mix_id += 1
            interpolation_examples.append([f'interpolation_{j}',mixed_tensor])
        embed[i] = interpolation_examples
    print(np.sum(lambdas))
    
    torch.save(embed,save_path)
    print("successfully saved")

#一次运行只生成一个文件，确保seed一致
interp_linear_beta(1,5,"./text_embed_linear_p_beta1_n5.bin","./text_embed.bin")
# interp_linear_beta(1,10,"./text_embed_linear_p_beta1_n10.bin","./text_embed.bin")
# interp_linear_beta(4,5,"./text_embed_linear_p_beta4_n5.bin","./text_embed.bin")
# interp_linear_beta(8,5,"./text_embed_linear_p_beta8_n5.bin","./text_embed.bin")
# interp_linear_beta(16,5,"./text_embed_linear_p_beta16_n5.bin","./text_embed.bin")
    


    