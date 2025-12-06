# Installation Steps

Follow the installation steps in order. This ensures that everything will be installed properly.

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```

***Make sure to be in the `llava` conda environment from now on in order to download the weights and run the code***

4. Make sure to not be in the LLaVA directory. Then, clone our repository.
```bash
cd ..
git clone https://github.com/llava-pokemon/pokemon_llava_fine_tune.git

cd pokemon_llava_fine_tune
```

5. Navigate to the data directory and download the images.
```bash
python download_images.py
```

6. **OPTIONAL:** If you want to fine-tune the model yourself, then you need to also run the following command while in the data directory.
```bash
python new_json.py
```
This is to follow the json structure that the fine-tuning script follows.

7. Go back to the root folder and download the weights to our model.
```bash
cd ..
huggingface-cli download Alterex/LLaVAFinetune_PokemonCards \
  --local-dir ./checkpoints/llava-v1.5-13b-lora \
  --local-dir-use-symlinks False
```

# Inference
To run an inference of our fine-tuned model, run the `inference.py` script.

Example usage:
```bash
python inference.py --query "What is this Pokemon's best attack?" --image https://images.pokemontcg.io/pl3/1_hires.png
```
```bash
python inference.py --image data/pokemon_images/xy8_160_hires.jpg
```

The script can take both a link to an image, or a path to an image.

### Dataset
To inference the whole dataset run the `inferenc_data.py` script.

Example usage:
```bash
python inference_dataset.py \
    --model-path checkpoints/llava-v1.5-13b-lora \
    --model-base liuhaotian/llava-v1.5-13b \
    --test-file data/pokemon_llava_dataset_test.jsonl \
    --image-folder data/pokemon_images \
    --query "What can this pokemon do in the Pokemon Trading Card game? Explain in detail." \
    --max_new_tokens 256 \
    --output-file finetune.jsonl
```

If you have two GPUs that you can use, then the following command will run the scripts on both GPUs. This will also create the base.jsonl and finetune.jsonl files:

```bash
./scripts/inference.sh
```

# Evaluation
To evaluate the the base and fine-tuned models using the LLM judges, two files are needed before running the `compare_judges.py` script:
* JSONL file that contains base model's evaluation
* JSONL file that contains fine-tuned models evaluation

This is created via the `inference_dataset.py` script. This script needs to be run twice with one using the base model and the other using the fine-tuned model. Ideally the JSONL files are named `base.jsonl` and `finetune.jsonl` for proper usage with other scripts.

Like how it was mentioned before, the `inference_dataset.sh` already creates both files and names them properly. 

Once these files are created, run the `judge_eval.py` script.

Example Usage:
```bash
python judge_eval.py \
  --base base.jsonl \
  --ft finetune.jsonl \
  --out qwen_judge.jsonl
```
```bash
python judge_eval.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --base base.jsonl \
  --ft finetune.jsonl \
  --out mistral_judge.jsonl 
```

The file names in the previous commands can be named something differently if preferred.

If you have two GPUs, then you can run the follow bash script:
```bash
./scripts/judge_eval.sh
```
Remember to have the file named properly in order to run this script, or change the file names in the script.

# Comparing Judges

To compare the LLM judges scores and winners, run the `compare_judges.py`. This utilizes the JSONL file that is created from `judge_eval.py`.

Example Usage:
```bash
python --model1 mistral_judge.jsonl --model2 qwen_judge.jsonl
```

This will output to terminal quantitative results from both judges as well as plots/figures. The plots and figures will be stored in `plots/` directory and create it if it isn't there already.

# OPTIONAL: Fine-Tuning
If you wish to fine-tune the model, then I would recommend that the `finetune_lora.sh` script gets moved to `LLaVA/scripts/v1_5/` and have it replace the `finetune_lora.sh` script that's already there.

Make sure to go into the script and change the following arguments to your path:
```
    --data_path /path/to/jsonl/file \
    --image_folder /path/to/image/folder \
```

The path to the data relative to the root directory would be `data/pokemon_llava_dataet_train.json` and the path to the images would be `data/pokemon_images`.

Then run the `finetune_lora.sh`.

Ideally, have at least two A100 80GB GPUs to not run into any problems when finetuning with these hyperparameters. If you have other hardware then you would need to change these hyperparameters:
```
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
```
Make sure to keep the batch size the same: `per_device_train_batch_size x gradient_accumulation_steps x num_gpus`.