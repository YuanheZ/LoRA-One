import torch
import os
import typing as tp
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PredictionOutput
from datasets import Dataset, load_dataset, load_metric
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from lora_plus import LoraPlusTrainingArguments, LoraPlusTrainer
from logTrainer import LogTrainer
import logging
import wandb
from peft import PeftModel
from data import load_alpaca

log = logging.getLogger(__name__)


def causalLMEncode(example, tokenizer, max_length=-1, ignore_masked_token=True):
    is_list_input = isinstance(example["x"], list)
    # Combine text and add EOS token
    combined_text = (
        [
            x + " " + y + tokenizer.eos_token
            for (x, y) in zip(example["x"], example["y"])
        ]
        if is_list_input
        else example["x"] + " " + example["y"] + tokenizer.eos_token
    )
    # Tokenize combined text
    encodings = tokenizer(
        combined_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length if max_length != -1 else None,
    )
    # Calculate input text length in tokens
    input_text_length = (
        [
            len(tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
            for i in range(len(example["x"]))
        ]
        if is_list_input
        else len(tokenizer(example["x"], return_tensors="pt")["input_ids"][0])
    )
    if input_text_length[0] >= max_length:
        log.warning(
            f"Input text length >= max_length: {input_text_length} >= {max_length}. "
            "Consider increasing max_length to avoid truncation."
        )
    # Create labels
    labels = encodings["input_ids"].clone()
    if is_list_input:
        for i, l in enumerate(input_text_length):
            labels[i, :l] = -100
    else:
        labels[0, :input_text_length] = -100
    if ignore_masked_token:
        labels[encodings["attention_mask"] == 0] = -100
    # Update example dictionary
    results = {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        # "input_text_length": input_text_length,
    }

    return results


def SeqToSeqEncode(example, tokenizer, max_length=None, ignore_masked_token=False):
    inputs = tokenizer(
        example["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    outputs = tokenizer(
        example["y"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    results = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
        "decoder_attention_mask": outputs["attention_mask"],
    }

    if ignore_masked_token:
        results["labels"][outputs["attention_mask"] == 0] = -100

    return results


def preprocess_dataset(
    dataset: tp.Union[Dataset, tp.List[tp.Tuple[str, str]], tp.List[tp.Dict[str, str]]]
) -> Dataset:
    if isinstance(dataset, list) and isinstance(dataset[0], tuple):
        dataset = Dataset.from_pandas(pd.DataFrame(dataset, columns=["x", "y"]))
    elif isinstance(dataset, list) and isinstance(dataset[0], dict):
        dataset = Dataset.from_dict(
            {k: [dic[k] for dic in dataset] for k in dataset[0]}
        )
    elif isinstance(dataset, dict):
        dataset = Dataset.from_dict(dataset)
    elif isinstance(dataset, Dataset):
        pass
    else:
        raise ValueError("Wrong format")
    return dataset


def initialize_text_to_text_model(
    model_name: str,
    model_type: str,
    bf16: bool,
    use_peft: bool = True,
    tokenizer: str = None,
    flash_attention: bool = False,
):
    if model_type == "CausalLM":
        if flash_attention:
            log.info("Using flash attention 2")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                #device_map="auto" if use_peft else None,
                attn_implementation="flash_attention_2",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if bf16 else torch.float32,
                #device_map="auto" if use_peft else None,
            )
    elif model_type == "ConditionalGeneration":
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if bf16 else torch.float32,
            device_map="auto" if use_peft else None,
        )
    if tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_metrics(p: PredictionOutput):
    predictions = p.predictions
    label_ids = p.label_ids # shape (batch_size, seq_len)
    if False:
        # Hard metric: the model must output exactly the same as the target
        # This should be the default evaluation metric for most tasks
        pred = np.argmax(predictions[0], axis=-1)
        num_correct = sum([np.array_equal(pred[i], label_ids[i]) for i in range(len(pred))])
        accuracy = num_correct / len(pred)
    else:
        # Soft metric: we limit the output space to the target space
        # i.e. the model classify the one with higher prob in positive and negative
        # **Use it in cola and mrpc, because it's too hard for vanilla lora**
        # Only suit for the binary classification with each label of 1 token
        label_ids = label_ids[:, 0] # remove the eos token
        unique_labels = np.unique(label_ids)
        flipped_labels = np.ones_like(label_ids) * unique_labels.sum() - label_ids
        predictions = predictions[0][:, 0, :] # remove the eos token # seq_len, tokens
        label_prob = predictions[np.arange(len(predictions)), label_ids]
        flipped_label_prob = predictions[np.arange(len(predictions)), flipped_labels]
        num_correct = sum(label_prob > flipped_label_prob)
        accuracy = num_correct / len(label_prob)

    return {"accuracy": accuracy}

'''def compute_metrics(p: PredictionOutput):
    predictions = p.predictions
    label_ids = p.label_ids # shape (batch_size, seq_len)
    pred = np.argmax(predictions[0], axis=-1)
    num_correct = sum([np.array_equal(pred[i], label_ids[i]) for i in range(len(pred))])
    accuracy = num_correct / len(pred)

    return {"accuracy": accuracy}'''


def transform_dataset(model_type, tokenizer, dataset, max_length):
    if model_type == "CausalLM":
        dataset.set_transform(lambda x: causalLMEncode(x, tokenizer, max_length))
    elif model_type == "ConditionalGeneration":
        dataset.set_transform(lambda x: SeqToSeqEncode(x, tokenizer, max_length))
    else:
        raise ValueError("Wrong model type")
    return dataset


def train_text_to_text_model(
    run_name: str,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    model_type: str,
    per_device_batch_size: int = 1,
    real_batch_size: int = 32,
    max_length: int = None,
    **kwargs,
) -> torch.nn.Module:
    # Preprocess the dataset
    train_dataset = preprocess_dataset(train_dataset)
    valid_dataset = preprocess_dataset(valid_dataset)

    assert (
        real_batch_size % per_device_batch_size == 0
    ), "real_batch_size must be divisible by per_device_batch_size"
    #accu_step = real_batch_size // per_device_batch_size
    accu_step = real_batch_size // (
        per_device_batch_size * kwargs.get("num_process", 1)
    )

    train_dataset, valid_dataset = transform_dataset(
        model_type, tokenizer, train_dataset, max_length
    ), transform_dataset(model_type, tokenizer, valid_dataset, max_length)

    eval_steps = (
        int(len(train_dataset) * kwargs.get("eval_epochs", 1)) // real_batch_size
    )
    # Special for loraplus
    use_loraplus = kwargs.get("use_loraplus", False)
    TrainingArgumentsClass = (
        LoraPlusTrainingArguments if use_loraplus else Seq2SeqTrainingArguments
    )
    TrainerClass = LoraPlusTrainer if use_loraplus else LogTrainer
    if use_loraplus:
        additional_kwargs = {
            "loraplus_lr_ratio": kwargs.get("loraplus_lr_ratio", 1.0),
        }
        log.info(
            f"Begin training using LoraPlusTrainer with additional kwargs: {additional_kwargs}"
        )
    else:
        additional_kwargs = {}
        log.info("Begin training using Seq2SeqTrainer")

    # Training arguments
    output_dir = f"./results/{run_name}/{kwargs.get('seed')}"
    training_args = TrainingArgumentsClass(
        output_dir=output_dir,  # output directory
        num_train_epochs=kwargs.get(
            "num_train_epochs", 3
        ),  # total number of training epochs
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accu_step,
        logging_dir="./logs",  # directory for storing logs
        logging_steps=kwargs.get("logging_steps", 10),  # when to print log
        bf16=kwargs.get("bf16", False),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", False),
        optim=kwargs.get("optim", "adamw_torch"),
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        #save_steps=eval_steps,
        #save_strategy="steps",
        save_strategy="epoch",
        #save_total_limit=1,  # No need for saving
        save_total_limit=kwargs.get("eval_epochs", 1),
        load_best_model_at_end=kwargs.get("load_best_model_at_end", False),
        metric_for_best_model=kwargs.get("metric_for_best_model", "eval_loss"),
        greater_is_better=kwargs.get("greater_is_better", False),
        do_eval=True,
        learning_rate=kwargs.get("learning_rate", 5e-5),
        remove_unused_columns=False,  # We tokenize the dataset on the fly
        eval_accumulation_steps=kwargs.get("eval_accumulation_steps", real_batch_size),
        label_names=[
            "labels"
        ],  # Peft are not compatible with HF's default label names yet
        # Ref: https://discuss.huggingface.co/t/eval-with-trainer-not-running-with-peft-lora-model/53286
        weight_decay = kwargs.get("weight_decay", 0), # No weight decay
        #warmup_ratio = 0.03,
        warmup_steps = 100,
        lr_scheduler_type = "linear", # cosine
        seed = kwargs.get("seed", 42),
        # Multi-GPU setting
        #dataloader_num_workers=4,
        #dataloader_pin_memory=True,
        #ddp_find_unused_parameters=False,  # Critical for LoRA on multi-GPU, ref: https://discuss.huggingface.co/t/training-llama-with-lora-on-multiple-gpus-may-exist-bug/47005
        **additional_kwargs,
    )

    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics if "llama" not in run_name else None,
        #callbacks=[
        #    EarlyStoppingCallback(
        #        early_stopping_patience=kwargs.get("early_stopping_patience", 10)
        #    ),
        #],
    )

    trainer.train()

    return model

def model_inference(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    input_text: str,
    model_type: str,
    max_source_length: str = 768,
    max_target_length: str = 256,
):
    if model_type == "CausalLM":
        inputs = tokenizer(
            input_text + " ",
            return_tensors="pt",
            max_length=max_source_length,
            truncation=True,
            return_token_type_ids=False,
        )
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                return_dict_in_generate=True,
                output_scores=False,
                max_new_tokens=max_target_length,
                eos_token_id=tokenizer.eos_token_id,
                #top_p=0.95,
                #temperature=0.8,
            )
        pred_text = tokenizer.decode(
            outputs.sequences[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        )
    elif model_type == "ConditionalGeneration":
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_target_length)
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_text


def load_peft_model(model, peft_path: str):
    peft_paths = [f"{peft_path}/{i}" for i in os.listdir(peft_path) if "merge" not in i]
    for peft_path in peft_paths:
        print(f"loading and merging from {peft_path}")
        model: PeftModel = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()
    return model


def test_train():
    # Example usage using emo dataset
    dataset = load_dataset("emo")
    label_map = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
    dataset = dataset.map(lambda e: {"x": e["text"], "y": label_map[e["label"]]})
    train_set = dataset["train"]
    test_set = dataset["test"]

    model_name = "t5-small"
    model_type = "ConditionalGeneration"
    model, tokenizer = initialize_text_to_text_model(model_name, model_type)

    model = train_text_to_text_model(
        train_set,
        test_set,
        model,
        tokenizer,
        model_type,
        num_train_epochs=1,
        per_device_batch_size=64,
        real_batch_size=64,
    )
    # Use the model for inference in the testset, print the first 10 examples
    for i in range(10):
        print("Input:", test_set[i]["x"])
        print("Target:", test_set[i]["y"])
        print(
            "Prediction:",
            model_inference(model, tokenizer, test_set[i]["x"], model_type),
        )
        print()


def test_llama_alpaca():
    model_name = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    peft_path = "results/llama-alpaca_alpaca/gradient-ArB2r-adam/0"
    model, tokenizer = initialize_text_to_text_model(model_name, model_type, True)
    model = load_peft_model(model, peft_path)
    _, _, test_set = load_alpaca()
    for i in range(10):
        print("Input:", test_set[i]["x"])
        # print("Target:", test_set[i]["y"])
        print(
            "Prediction:",
            model_inference(model, tokenizer, test_set[i]["x"], model_type),
        )
        print()


def merge_llama(peft_path):
    model_name = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(model_name, model_type, True)
    model = load_peft_model(model, peft_path)
    print("Save model to ", os.path.join(peft_path, "merged_checkpoint"))
    model.save_pretrained(os.path.join(peft_path, "merged_checkpoint"))
    tokenizer.save_pretrained(os.path.join(peft_path, "merged_checkpoint"))
    del model, tokenizer

def merge_t5(peft_path):
    model_name = "t5-base"
    model_type = "ConditionalGeneration"
    model, tokenizer = initialize_text_to_text_model(model_name, model_type, True)
    model = load_peft_model(model, peft_path)
    print("Save model to ", os.path.join(peft_path, "merged_checkpoint"))
    model.save_pretrained(os.path.join(peft_path, "merged_checkpoint"))
    tokenizer.save_pretrained(os.path.join(peft_path, "merged_checkpoint"))
    del model, tokenizer

def merge_llama_code(peft_path):
    model_name = "meta-llama/Llama-2-7b-hf"
    model_type = "CausalLM"
    model, tokenizer = initialize_text_to_text_model(model_name, model_type, True)
    model = load_peft_model(model, peft_path)
    print("Save model to ", os.path.join(peft_path, "merged_checkpoint"))
    model.save_pretrained(os.path.join(peft_path, "merged_checkpoint"), safe_serialization=False)
    tokenizer.save_pretrained(os.path.join(peft_path, "merged_checkpoint"))
    del model, tokenizer


if __name__ == "__main__":
    merge_llama("results/llama-alpaca_alpaca/default/0")