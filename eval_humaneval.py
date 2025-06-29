from human_eval.data import write_jsonl, read_problems
from fire import Fire
from tqdm import trange, tqdm
from utils import initialize_text_to_text_model, model_inference
import re
import os

ALPACA_PREFIX_TEMPLATE_MD = """Below is an instruction that describes a task.\n Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code: 
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""

def post_process(text):
    text = text.replace("```", "")
    text = text.replace("\t", "    ")
    text = re.sub(r'(""".*?"""|\'\'\'.*?\'\'\')', '', text, flags=re.DOTALL)
    text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    lines = text.split("\n")
    spaces_for_each_line = []
    for line in lines:
        match = re.match(r'^( *)', line)
        if match:
            leading_spaces = len(match.group(1))
            spaces_for_each_line.append(leading_spaces)
    try:
        def_line = [i for i, line in enumerate(lines) if "def" in line][0]
        def_line_space = spaces_for_each_line[def_line]
    except:
        print("No def line found")
        print(text)
        def_line_space = 0
    rank_unique_spaces = sorted(list(set(spaces_for_each_line)))
    indentation_level = {}
    i = 0
    for space in rank_unique_spaces:
        if space <= def_line_space:
            indentation_level[space] = 0
        else:
            i += 1
            indentation_level[space] = i
    new_lines = []
    for line, space in zip(lines, spaces_for_each_line):
        new_lines.append("    " * indentation_level[space] + line.lstrip())
    return "\n".join(new_lines)

def generate_one_completion(model, tokenizer, model_type, prompt, template=True):
    if template:
        prompt_in = ALPACA_PREFIX_TEMPLATE_MD.format(PROMPT=prompt)
    pred_text = model_inference(model, tokenizer, prompt_in, model_type, max_target_length=512)
    post_pred = post_process(pred_text)
    return post_pred

def main(model_name, num_sample=1):
    best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}
    problems = read_problems()
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, True, tokenizer="meta-llama/Llama-2-7b-hf",flash_attention=True
    )
    #model.generation_config.temperature = best_temperature[num_sample]
    #model.generation_config.top_p = 0.95
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.do_sample = False
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(model, tokenizer, model_type, problems[task_id]["prompt"]))
        for task_id in tqdm(problems, desc="Tasks")
        for _ in range(num_sample)
    ]
    target_name = os.path.join("humaneval_samples", f"{model_name.replace('/', '_')}_nsamples{num_sample}_humaneval_samples.jsonl")
    write_jsonl(target_name, samples)

if __name__ == "__main__":
    Fire(main)
    
if __name__ == "__main__":
    Fire(main)
