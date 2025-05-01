from data import load_gsm8k
from utils import model_inference, initialize_text_to_text_model
from fire import Fire
import re
import os
from tqdm import tqdm
import wandb

def extract_num(text):
    # Regex pattern to find the number following '####'
    pattern = r'####\s*(\d+)'
    # Using re.search to find the first match
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        print(text)
        result = ""
    try:
        return int(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return 0

def main(model_name, wandb_name, temperature=None, top_p=None):
    wandb.init(
        entity="xxx",
        project="llama_eval_gsm8k",
        name=wandb_name
    )
    _, _, test_set = load_gsm8k()
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        model_name, model_type, True, tokenizer="meta-llama/Llama-2-7b-hf",flash_attention=True
    )
    model = model.to('cuda')
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p

    if model.generation_config.temperature != None:
       model.generation_config.do_sample=True
    else:
       model.generation_config.do_sample=False

    print("do sampling", model.generation_config.do_sample, "temperature", model.generation_config.temperature, "top_p", model.generation_config.top_p)
    all = 0
    correct = 0
    t = tqdm(test_set)
    for example in t:
        pred_text = model_inference(model, tokenizer, example['x'], model_type, max_target_length=512)
        gt = extract_num(example["y"])
        print('gt', gt)
        pred = extract_num(pred_text)
        print('pred', pred)
        correct += int(gt==pred)
        all += 1
        t.set_description(f"Accuracy: {correct/all*100:02f}%")
        wandb.log(
            {
                "Acc": correct/all,
            }
        )
        
    print("Acc:", correct/all)
    wandb.log(
        {
            "final_Acc": correct/all,
        }
    )
    # append to gsm8k_results.txt (create if not exists)
    if not os.path.exists("gsm8k_results.txt"):
        with open("gsm8k_results.txt", "w") as f:
            f.write("Model Acc\n")
    with open("gsm8k_results.txt", "a") as f:
        f.write(f"{model_name} {correct/all}\n")

if __name__ == "__main__":
    Fire(main)
