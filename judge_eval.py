import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import argparse
import re
import ast

JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"


# ============================================================================
# JSON EXTRACTION LOGIC
# ============================================================================
def extract_first_json_like(text: str):
    """
    Extract and parse the *first* JSON-like object found in a model's output.
    
    Why this is needed:
    -------------------
    LLMs often:
      - prepend or append natural language explanations
      - output multiple JSON blocks
      - produce JSON with minor formatting errors
      - include stray characters before the JSON starts

    This function isolates the first {...} block using a brace-balanced regex,
    then tries two parsing strategies:

      1. Strict json.loads (preferred, guarantees valid JSON)
      2. ast.literal_eval fallback (to recover from minor issues like single quotes)

    If no JSON block is found OR parsing truly fails, we raise a ValueError.
    """
    
    # Regex: captures the first "balanced" JSON-like {...} structure
    json_regex = r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}"
    match = re.search(json_regex, text, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON-like object found in model output:\n{text}")

    json_str = match.group(0).strip()

    # Strict JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback for slightly invalid JSON structures
        try:
            parsed = ast.literal_eval(json_str)
            if not isinstance(parsed, dict):
                raise ValueError(f"Parsed object is not a dict: {parsed}")
            return parsed
        except Exception as e:
            raise ValueError(
                f"Failed to parse JSON-like object.\nString:\n{json_str}\nError: {e}"
            )


# ============================================================================
# JUDGE PROMPT CONSTRUCTION
# ============================================================================
def build_judge_prompt(example):
    """
    Construct the prompt given to the judgment LLM.
    
    High-level design goals:
    ------------------------
    - Force the model into JSON-only output mode
    - Clearly delineate Candidate A and Candidate B with markers
    - Strict constraints to avoid "helpful" natural language output
    - Make the LLM's job binary: compare two answers, output structured JSON
    
    Strictness is necessary because:
      - Some LLMs like to "explain" first
      - Some prepend apologies or clarifications
      - Some output multiple JSON blocks unless explicitly forbidden
    """
    a_base = example["model_base_answer"]
    a_ft = example["model_ft_answer"]

    prompt = f"""
You are a STRICT JSON-ONLY judge for two answers (A and B) about the same Pokémon TCG card.

Your job:
1. Decide which answer is better overall: "A", "B", or "Tie".
2. Give an integer score from 1 to 10 for each answer based on:
   - Level of detail
   - How plausible and visually grounded it seems
   - Clarity and coherence

Here are the two candidate answers:

[START_CANDIDATE_A]
{a_base}
[END_CANDIDATE_A]

[START_CANDIDATE_B]
{a_ft}
[END_CANDIDATE_B]

Now, based on these two answers, respond with EXACTLY ONE JSON OBJECT and NOTHING ELSE.

OUTPUT FORMAT (VERY IMPORTANT):
- Your ENTIRE reply must be EXACTLY ONE JSON object.
- The FIRST character of your reply must be "{{".
- The LAST character of your reply must be "}}".
- Do NOT write any text before or after the JSON.
- Do NOT wrap the JSON in backticks.
- Use DOUBLE QUOTES for all keys and string values.
- Do NOT use single quotes.
- Do NOT add trailing commas.
- Do NOT add any extra keys.

The JSON object MUST have EXACTLY these fields:
{{
  "winner": "A" or "B" or "Tie",
  "A_score": <integer from 1 to 10>,
  "B_score": <integer from 1 to 10>,
  "rationale": "<brief explanation (1-3 sentences)>"
}}

Ties can only be chosen if "A_score" and "B_score" are identical.

Respond NOW with ONLY the JSON object in this format.
""".strip()

    return prompt


# ============================================================================
# RUN JUDGE MODEL ON ONE PAIR OF ANSWERS
# ============================================================================
def judge_example(model, tokenizer, example, max_new_tokens=128):
    """
    Pass the two candidate answers through the judge LLM.
    
    This function handles:
      - Prompt creation
      - Tokenization + GPU dispatch
      - Model generation
      - JSON extraction + schema validation
      - Fallback logic when JSON is incomplete
    
    By keeping this step isolated, the rest of the pipeline remains modular.
    """
    prompt = build_judge_prompt(example)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    # Run judge model deterministically (no sampling)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0,   # fully deterministic judging
            do_sample=False
        )

    # Slice away the prompt and decode only model's continuation
    gen_tokens = out[0][input_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Try to recover a JSON dict
    res = extract_first_json_like(text)

    # Robust fallback: ensures script never crashes
    if res is None:
        return {
            "winner": "Tie",
            "A_score": 5,
            "B_score": 5,
            "rationale": "Judge failed to produce valid JSON."
        }, text

    # Required keys check
    required = ["winner", "A_score", "B_score", "rationale"]
    if not all(k in res for k in required):
        print("[WARN] Parsed object missing keys. Parsed:", res, "\nRaw output:\n", text[:500], "...\n", flush=True)
        return {
            "winner": "Tie",
            "A_score": 5,
            "B_score": 5,
            "rationale": "Judge produced incomplete JSON."
        }, text

    return res, text


# ============================================================================
# MAIN BATCH EVALUATION LOOP
# ============================================================================
def main(model_name, base_path, ft_path, out_path):
    """
    Evaluate a *paired* dataset of base-model and finetuned-model outputs.
    
    Pipeline:
    ---------
    1. Load judge model (Qwen or Mistral)
    2. Iterate over base/ft jsonl files line-by-line in sync
    3. For each entry:
         - Build example pair
         - Run judge LLM
         - Save structured evaluation record
    4. Write results incrementally (streaming) to avoid memory pressure
    
    This is a streaming evaluator, suitable for large datasets (3k+ examples).
    """
    base_path = Path(base_path)
    ft_path = Path(ft_path)

    # Load judge tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # automatically distribute across GPUs if available
    )
    
    # Ensure model has a valid pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Begin output streaming
    with open(out_path, "w", encoding="utf-8") as f:
        with base_path.open("r") as fb, ft_path.open("r") as ff:

            for base_line, ft_line in tqdm(
                zip(fb, ff),
                desc="Judging examples",
                unit="_entry"
            ):
                # Exit if files become unsynchronized
                if not base_line or not ft_line:
                    break

                base_ex = json.loads(base_line)
                ft_ex = json.loads(ft_line)

                # Safety check: matching image keys
                img_base = base_ex.get("image")
                img_ft = ft_ex.get("image")
                if img_base != img_ft:
                    print(f"[WARN] Image mismatch: {img_base} vs {img_ft}")

                # Prepare data to judge (no need to pass image string)
                ex = {
                    "model_base_answer": base_ex.get("output", ""),
                    "model_ft_answer": ft_ex.get("output", ""),
                }

                # Output dict begins with image ID
                final = {
                    "image": img_base
                }

                # Run judge model
                judge_res, raw = judge_example(model, tokenizer, ex)

                # Merge results safely
                final.update(judge_res)
                
                # Write immediately (important for long runs)
                json.dump(final, f, ensure_ascii=False)
                f.write("\n")
                f.flush()   # prevents data loss on crash


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=JUDGE_MODEL)
    ap.add_argument("--base", required=True, help="Base model outputs")
    ap.add_argument("--ft", required=True, help="Finetuned model outputs")
    ap.add_argument("--out", required=True, help="Judged outputs")
    args = ap.parse_args()

    main(args.model, args.base, args.ft, args.out)
