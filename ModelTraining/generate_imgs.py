from datasets import load_dataset
from torch.utils.data import DataLoader,Dataset
import json
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=1)
    parser.add_argument("--device",type=int,default=2)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--max_num_imgs",type=int,default=1000)
    parser.add_argument("--model_name",type=str,default="coco-model")
    parser.add_argument("--output_dir",type=str,default="./coco-model_inference")
    parser.add_argument("--input_dir",type=str,default="./metadata_test.jsonl")
    parser.add_argument("--mini",type=bool,default=False)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device=torch.device("cuda:"+str(args.device))

    class TextDataset(Dataset):
        def __init__(self,data_dir=args.input_dir):
            self.texts = []
            self.imgs = []

            with open(data_dir) as fin:
                num_imgs = 0
                last_img_name = ""
                mini=100
                for line in fin.readlines():

                    dic=json.loads(line)
                    if dic["file_name"][:12]!=last_img_name:
                        last_img_name = dic["file_name"][:12]
                        num_imgs+=1
                    
                    if num_imgs > args.max_num_imgs:
                        break

                    if args.mini:
                        if dic["file_name"][13]=='0':
                            self.texts.append(dic["text"])
                            self.imgs.append(dic["file_name"])
                            mini-=1
                        if mini==0:
                            break        
                    else:
                        self.texts.append(dic["text"])
                        self.imgs.append(dic["file_name"])
                        

        def __getitem__(self,item):
            return self.texts[item],self.imgs[item]
        def __len__(self):
            return len(self.texts)


    eval_dataloader=DataLoader(TextDataset(),batch_size=args.batch_size,shuffle=False)
    pipe = StableDiffusionPipeline.from_pretrained(args.model_name)
    pipe.safety_checker=None
    pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    for texts,image_names in tqdm(eval_dataloader):
        with torch.autocast("cuda"):
            images=pipe(list(texts)).images
        for name,img in zip(image_names,images):
            img.save(args.output_dir+"/" + name)

if __name__=="__main__":
    main()









