import os
import re

walkthrough_path = '/home/levos/experiments/WALKTHROUGH.md'
images_dir = '/home/levos/experiments'

# Find all pngs
all_pngs = []
for root, dirs, files in os.walk(images_dir):
    if '.ipynb_checkpoints' in root:
        continue
    for f in files:
        if f.endswith('.png'):
            all_pngs.append(os.path.join(root, f))

with open(walkthrough_path, 'r') as f:
    lines = f.readlines()

fixed_lines = []

for idx, line in enumerate(lines):
    stripped = line.strip()
    # If the line is just a single word or a few words that might be an image name
    # e.g. "training_metrics" or "Event 0 Comparison"
    
    # Check if it matches a known image name without extension
    # We will check if `stripped` (with spaces replaced by underscores, or just as is) matches any png
    
    matched = False
    if len(stripped) > 2 and not stripped.startswith('#') and not stripped.startswith('-') and not stripped.startswith('|'):
        # Try to find matching png
        # 1. Exact match with .png
        # 2. Match with spaces replaced by underscores
        possible_names = [
            f"{stripped}.png",
            f"{stripped.replace(' ', '_')}.png",
            f"{stripped.replace(' ', '')}.png",
            f"{stripped.lower().replace(' ', '_')}.png"
        ]
        
        for p in all_pngs:
            basename = os.path.basename(p)
            if basename in possible_names:
                rel_path = p.replace('/home/levos/experiments/', '')
                indent = line[:len(line) - len(line.lstrip())]
                fixed_lines.append(f"{indent}![{stripped}]({rel_path})\n")
                print(f"Line {idx+1}: Fixed {stripped} -> {rel_path}")
                matched = True
                break
                
    if not matched:
        fixed_lines.append(line)

with open(walkthrough_path, 'w') as f:
    f.writelines(fixed_lines)

print("Done fixing images.")
