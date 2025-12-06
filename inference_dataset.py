import argparse
import torch
import os
import json
from PIL import Image
import requests
from io import BytesIO
import re

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)





def load_image(image_file):
    """
    Load a single image from either a URL or local file path.
    
    Args:
        image_file: Path to local image or URL starting with http/https
        
    Returns:
        PIL Image object in RGB format
    """
    # Check if the image is hosted online
    if image_file.startswith("http") or image_file.startswith("https"):
        # Download the image from the URL
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # Load the image from local file system
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    """
    Load multiple images from a list of file paths or URLs.
    
    Args:
        image_files: List of image paths or URLs
        
    Returns:
        List of PIL Image objects
    """
    out = []
    # Iterate through each image file and load it
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    """
    Run inference on a dataset using a pretrained LLaVA model.
    
    Args:
        args: Command-line arguments containing model path, test file, query, etc.
    """
    # Model initialization
    # Disable PyTorch default initialization for faster loading
    disable_torch_init()

    # Extract the model name from the model path
    model_name = get_model_name_from_path(args.model_path)
    # Load the pretrained model, tokenizer, image processor, and get context length
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # Prepare the query string with image tokens
    qs = args.query
    # Create image token sequence with start and end tokens
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # Check if the query contains an image placeholder
    if IMAGE_PLACEHOLDER in qs:
        # Replace placeholder with appropriate image token(s) based on model config
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        # If no placeholder, prepend image token(s) to the query
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    # Determine the conversation mode based on the model name
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # Check if user specified a different conversation mode than auto-inferred
    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    # Create a conversation template and build the prompt
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)

    # Get the formatted prompt string
    prompt = conv.get_prompt()

    # Load the test dataset (JSONL format - one JSON object per line)
    with open(args.test_file, "r", encoding="utf-8") as f:
        test = [json.loads(line) for line in f]

    # Get total number of test samples and print for progress tracking
    test_length = len(test)
    print(f"Total test samples: {test_length}")

    # Process each test sample
    for i, entry in enumerate(test):
        # Get the image filename from the test entry
        image_file = entry["image"]
        # Construct full path to the image file
        full_image_path = os.path.join(args.image_folder, image_file)
        # Load the image
        images = load_images([full_image_path])
        # Get image dimensions for processing
        image_sizes = [images[0].size]
        # Process images into tensor format and move to GPU with float16 precision
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)

        # Tokenize the prompt with image tokens and prepare for model input
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

        # Run inference without gradient computation for efficiency
        with torch.inference_mode():
            # Generate model output using the prepared inputs
            output_ids = model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,  # Enable sampling if temperature > 0
                temperature=args.temperature,  # Controls randomness in generation
                top_p=args.top_p,  # Nucleus sampling parameter
                num_beams=args.num_beams,  # Number of beams for beam search
                max_new_tokens=args.max_new_tokens,  # Maximum length of generated text
                use_cache=True,  # Use KV cache for faster generation
            )

        # Decode the generated token IDs back to text
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        # Append the result to the output file in JSONL format
        with open(args.output_file, "a", encoding="utf-8") as f:
            json.dump({
                "image": image_file,
                "output": outputs
            }, f)
            f.write("\n")

        # Print progress update every 10 samples or at the end
        if (i + 1) % 10 == 0 or (i + 1) == test_length:
            print(f"Processed {i + 1}/{test_length} samples.")
    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="output.json")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    # Run the evaluation
    eval_model(args)
