import json

input_path = "pokemon_llava_dataset_train.jsonl"
output_path = "pokemon_llava_dataset_train_fixed.json"

# Read all raw lines
with open(input_path, "r", encoding="utf-8") as f:
    lines = [l.rstrip("\n") for l in f]

# Strip whitespace-only lines
lines = [l for l in lines if l.strip()]

# Must begin with '[' and end with ']'
if not lines[0].strip().startswith('['):
    raise RuntimeError("First line must contain '['")
if not lines[-1].strip().endswith(']'):
    raise RuntimeError("Last line must contain ']'")

# Remove the opening '[' and closing ']' for processing
content_lines = lines[1:-1]

fixed_lines = []

for i, line in enumerate(content_lines):
    stripped = line.strip()

    # Remove trailing commas if any
    if stripped.endswith(','):
        stripped = stripped[:-1]

    # Add a comma to every line EXCEPT the last
    if i < len(content_lines) - 1:
        fixed_lines.append(stripped + ',')
    else:
        fixed_lines.append(stripped)

# Reconstruct final JSON
final_json_text = "[\n" + "\n".join(fixed_lines) + "\n]"

# Validate JSON
try:
    parsed = json.loads(final_json_text)
    print(f"Parsed JSON successfully: {len(parsed)} items")
except Exception as e:
    raise RuntimeError(f"Resulting JSON is invalid: {e}")

# Write to output
with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_json_text)

print(f"Wrote fixed JSON to: {output_path}")
