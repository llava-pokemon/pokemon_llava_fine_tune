from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import argparse

prompt = "In terms of the Pokemon Trading Card game, what makes this card standout compared to other cards? Also what set is this from? Which pokemon would be really weak against this card?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=prompt)
    ap.add_argument("--image", required=True, help="The image file path or URL")

    args = ap.parse_args()

    args1 = type('Args', (), {
    "model_path": "checkpoints/llava-v1.5-13b-lora",
    "model_base": "liuhaotian/llava-v1.5-13b",
    "query": args.query,
    "conv_mode": None,
    "image_file": args.image,
    "sep": ",",
    "temperature": 0.8,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 1024
    })()
    
    eval_model(args1)
