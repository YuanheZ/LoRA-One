# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy
import json
import os
import re
import sys
import argparse
from utils import (
    initialize_text_to_text_model,
)
import fire
import torch

#sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import wandb

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main():
    args = parse_args()

    def evaluate(
            instructions,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=32,
            **kwargs,
    ):
        prompts = [generate_prompt(instruction, input) for instruction in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            do_sample=True,
            max_length=512, #remember to change
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        greedy_config = GenerationConfig(
            do_sample=False,
            max_length=512, #remember to change
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=4,
            **kwargs,
        )
        with torch.no_grad():
          try: #fix generation issue
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
          except RuntimeError as e:
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=greedy_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]
        return outputs

    save_file = f'comm_eval/{args.name}.txt'
    create_dir('comm_eval/')

    wandb.init(
        entity="yuanhezhang6-university-of-warwick",
        project="comm_eval",
        name=args.name
    )

    dataset = load_data(args)
    batches = create_batch(dataset, args.batch_size)
    
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(
        args.base_model, model_type, True, tokenizer="meta-llama/Llama-2-7b-hf",flash_attention=False
    )
    model = model.to('cuda')
    #new features
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.half()

    total = len(batches)
    correct = 0
    current = 0
    output_data = []
    pbar = tqdm(total=total)
    for idx, batch in enumerate(batches):
        current += len(batch)
        instructions = [data.get('instruction') for data in batch]

        outputs = evaluate(instructions)

        for data, output in zip(batch, outputs):
            label = data.get('answer')
            flag = False
            predict = extract_answer(args, output)
            if label == predict:
                correct += 1
                flag = True
            new_data = copy.deepcopy(data)
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            print(data["instruction"])
            print(output)
            print('prediction:', predict)
            print('label:', label)
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        wandb.log(
            {
                f"{args.dataset}/Acc": correct / current,
            }
        )
        pbar.update(1)
    pbar.close()
    '''final_output = {
                f"{args.dataset}": correct / current,
    }'''
    '''with open(save_file, 'w+') as f:
        json.dump(final_output, f, indent=4)'''
    if not os.path.exists(save_file):
        with open(save_file, "w") as f:
            f.write("Model Acc\n")
    with open(save_file, "a") as f:
        f.write(f"{args.dataset} {correct / current}\n")
    print('\n')
    print('test finished')

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    return

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''

def generate_prompt(instruction, input=None):
    return template_wo_input.format(instruction=instruction)

def load_data(args) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{args.dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data

def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"],
                        required=True)
    parser.add_argument('--base_model', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    return parser.parse_args()

def load_instruction(args) -> str:
    instruction = ''
    if not instruction:
        raise ValueError('instruct not initialized')
    return instruction


def extract_answer(args, sentence: str) -> float:
    dataset = args.dataset
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


if __name__ == "__main__":
    main()
