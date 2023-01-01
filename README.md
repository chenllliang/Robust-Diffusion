Source code for project report "On The Robustness of Diffusion-Based Text-to-Image Generation" in CV-2022-Fall.

# Model Training

```
cd ModelTraining
```

## Data Preparing
Before doing this experiment, please download images of _MSCOCO 2017_ dataset from https://cocodataset.org/#download

## Environment Setting
We use the same environment as _stable-diffusion_ https://github.com/CompVis/stable-diffusion 
A suitable conda environment named `ldm` can be created and activated with: 

```
conda env create -f environment.yaml
conda activate ldm
```

## Model Training
Specify which GPU (or GPUs) you want to use to train the model with:
```
accelerate config
```
Set proper hyper-parameters in `tune.sh`. Do remember to set _train_data_dir_ to the directory of your training set ! 
Training the model with:
```
bash tune.sh
```

If you want to use text augment methods like _back translation_,_text crop and swap_, you can add this augment to `tune.sh`:
```
--text_augment="bt" or --text_augment="crop_swap"
```

If you want to use text interpolation augment method to train the model, you can first run `python encode_text.py` to generate interpolation text vectors, and then add `--text_embed_dir="./text_embed_linear_p_beta1_n5.bin"`to `tune.sh`

## inference
After training, we can generate images with trained model under the control of the texts in test set. You can generate images with:
```
bash generate.sh
```
Before generating images, set proper hyper-parameters in `generate.sh`: `--model_name` is the name of directory of your trained model; `--output_dir` is the directory of generated images. 


# Interpolation

```
cd Interpolation
```

## Text Data Augmentation Method Implementation

### Hidden States Interpolation

Note that this method needs the hidden states of text after clip encoder.

```bash
python HiddenStatesInterpolation.py
```

### Other Augmentation Method (Random Deleting and Back-Translation)

we provide a jupyter notebook, please run `Interpolation.ipynb`

# RobustnessAnalysis

```
cd RobustnessAnalysis
```

1. The first similarity "**Image Similarity among random seeds** " is in "Similarity_with_seed.ipynb", and it's similarity between the images generated from the same texts but with different seeds.
2. The second similarity "**Similarity2: within similar texts**" is in "similarity.ipynb", and its Chinese name is"组内相似度".
3. The third similarity "**Faithfulness : between image and text**" is in "text_img_similarity.ipynb", and it's similarity between  the images and the texts.