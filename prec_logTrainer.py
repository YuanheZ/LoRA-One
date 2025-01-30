from dataclasses import dataclass, field
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim.optimizer import Optimizer
import math

from transformers import Trainer, Seq2SeqTrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled, logging
from peft.tuners.lora.layer import Linear as LoraLinear

# include_keywords = ["block.0", "block.4"]
# include_keywords = ["encoder.block.2","encoder.block.3","encoder.block.4"]  # for T5
include_keywords = ["layers.27", "layers.6"]  # for Llama
do_log = False

# https://github.com/pilancilab/Riemannian_Preconditioned_LoRA.git
class AdamWr(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, correct_bias=False, optimizer_reg=1e-6):
        defaults = dict(lr=lr, betas=betas, eps=eps, correct_bias=correct_bias)
        super().__init__(params, defaults)
        self.reg = optimizer_reg
        print('Using Optimizer Scaled AdamW') 
        print('learning rate: ', lr)
        print('betas: ', betas)
        print('reg: ', self.reg)
    def reset_state(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p1, p2 in list(zip(group["params"],group["params"][1:]))[::2]:
                state = self.state[p1]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p1.data)
                    state["exp_avg_sq"] = torch.zeros_like(p1.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                grad1 = p1.grad.data
                c = p2.data
                try:
                    c_ = torch.inverse(c.T@c+self.reg*torch.eye(c.shape[1]).to(c.device))
                except:
                    c_ = torch.eye((c.T@c).shape[0]).to(c.device)
                    
                grad1_scaled = c_@grad1
                assert grad1_scaled.shape == p1.grad.data.shape

                exp_avg.mul_(beta1).add_(grad1_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad1_scaled, grad1_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                c1 = p1.data

                p1.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p1.data.add_(p1.data, alpha=-group["lr"] * group["weight_decay"])

                
                state = self.state[p2]
                
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p2.data)
                    state["exp_avg_sq"] = torch.zeros_like(p2.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                grad2 = p2.grad.data
                try:
                    c1_ = torch.inverse(c1@c1.T+self.reg*torch.eye(c.shape[1]).to(c.device))
                except:
                    c1_ = torch.eye((c1@c1.T).shape[0]).to(c1.device)
                    
                
                grad2_scaled = grad2@c1_
                assert grad2_scaled.shape == p2.grad.data.shape
                
                exp_avg.mul_(beta1).add_(grad2_scaled, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad2_scaled, grad2_scaled, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if 'correct_bias' in group and group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p2.data.addcdiv_(-step_size, exp_avg, denom)
                if group["weight_decay"] > 0.0:
                    p2.data.add_(p2.data, alpha=-group["lr"] * group["weight_decay"])
                
        return loss


def get_forward_hook(name):
    def hook(module, input, output):
        wandb.log(
            {
                f"{name}/input_mean": input[0].mean().item(),
                f"{name}/input_std": input[0].std().item(),
                f"{name}/output_mean": output.mean().item(),
                f"{name}/output_std": output.std().item(),
            },
            commit=False,
        )
    return hook

class LogTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: Seq2SeqTrainingArguments = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.is_peft = "PeftModel" in type(model).__name__
        if self.is_peft:
            for name, module in model.named_modules():
                if isinstance(module, LoraLinear):
                    self.scaling = module.scaling["default"]
                    break
        self.orig_A = None
        self.orig_B = None
        self.orig_W = None
        self.gradient_accumulation_counter = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        if not do_log:
            return super().training_step(model, inputs)
        if self.is_peft:
            if self.orig_A is None:
                self.orig_A = {}
                self.orig_B = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        if "lora_A" in name:
                            self.orig_A[name.split("lora_A.")[0]] = (
                                param.detach().clone()
                            )
                        elif "lora_B" in name:
                            self.orig_B[name.split("lora_B.")[0]] = (
                                param.detach().clone()
                            )
                '''for name, module in model.named_modules():
                    if any([kw in name for kw in include_keywords]) and isinstance(module, LoraLinear):
                        breakpoint()
                        hook = get_forward_hook(name)
                        module.register_forward_hook(hook)'''
        else:
            if self.orig_W is None:
                self.orig_W = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and any(
                        [kw in name for kw in include_keywords]
                    ):
                        self.orig_W[name] = param.detach().clone()

        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        with torch.no_grad():
            if (
                self.gradient_accumulation_counter
                % self.args.gradient_accumulation_steps
                == self.args.gradient_accumulation_steps - 1
            ):
                if self.is_peft:
                    A_dict = {}
                    B_dict = {}
                    for name, param in model.named_parameters():
                        if param.requires_grad and any(
                            [kw in name for kw in include_keywords]
                        ):
                            if "lora_A" in name:
                                A_dict[name.split("lora_A.")[0]] = param
                            elif "lora_B" in name:
                                B_dict[name.split("lora_B.")[0]] = param
                    assert (
                        len(A_dict)
                        == len(self.orig_A)
                        == len(B_dict)
                        == len(self.orig_B)
                    ), (
                        len(A_dict),
                        len(self.orig_A),
                        len(B_dict),
                        len(self.orig_B),
                    )
                    for key in A_dict.keys():
                        A = A_dict[key]
                        B = B_dict[key]
                        lora_r = A.shape[0]
                        A_grad = A_dict[key].grad
                        B_grad = B_dict[key].grad
                        A_0 = self.orig_A[key]
                        B_0 = self.orig_B[key]
                        A_diff = A - A_0
                        B_diff = B - B_0
                        BA = torch.matmul(B, A)
                        BA_0 = torch.matmul(B_0, A_0)
                        BA_diff = BA - BA_0
                        BA_diff_norm = torch.norm(BA_diff).item()
                        A_diff_norm = torch.norm(A_diff).item()
                        B_diff_norm = torch.norm(B_diff).item()
                        A_norm = torch.norm(A).item()
                        B_norm = torch.norm(B).item()
                        A_grad_norm = torch.norm(A_grad).item()
                        B_grad_norm = torch.norm(B_grad).item()
                        # BA_singular_values = torch.svd(BA_diff.float(), compute_uv=False).S[:lora_r]
                        BA_singular_values = torch.svd_lowrank(
                            BA_diff.float(), q=2 * lora_r
                        )[1][:lora_r]
                        top_1_ratio = (
                            BA_singular_values[0] / BA_singular_values.sum()
                        ).item()
                        top_4_ratio = (
                            BA_singular_values[:4].sum() / BA_singular_values.sum()
                        ).item()
                        wandb.log(
                            {
                                f"A_norm/{key}": A_norm,
                                f"B_norm/{key}": B_norm,
                                f"A_grad_norm/{key}": A_grad_norm,
                                f"B_grad_norm/{key}": B_grad_norm,
                                f"A_diff_norm/{key}": A_diff_norm,
                                f"B_diff_norm/{key}": B_diff_norm,
                                f"BA_diff_norm/{key}": BA_diff_norm,
                                f"scaled_BA_diff_norm/{key}": self.scaling
                                * BA_diff_norm,
                                f"BA_top_1_ratio/{key}": top_1_ratio,
                                f"BA_top_4_ratio/{key}": top_4_ratio,
                                #"train/global_step": self.state.global_step,
                            }
                        )

                        for idx in range(lora_r):
                          wandb.log(
                            {
                              f"BA_SV_{key}/idx_{idx+1}": BA_singular_values[idx].item()
                            }
                          )
                else:
                    W_dict = {}
                    for name, param in model.named_parameters():
                        if (
                            param.requires_grad
                            and any([kw in name for kw in include_keywords])
                            and len(param.shape) == 2
                        ):
                            W_dict[name] = param
                    for key in W_dict.keys():
                        W = W_dict[key]
                        W_grad = W.grad
                        W_0 = self.orig_W[key]
                        W_diff = W - W_0
                        W_diff_norm = torch.norm(W_diff).item()
                        W_norm = torch.norm(W).item()
                        W_grad_norm = torch.norm(W_grad).item()
                        U, S, V = torch.svd(W_diff.float())
                        top_1_ratio = S[0] / S.sum()
                        top_4_ratio = S[:4].sum() / S.sum()
                        wandb.log(
                            {
                                f"W_norm/{key}": W_norm,
                                f"W_grad_norm/{key}": W_grad_norm,
                                f"W_diff_norm/{key}": W_diff_norm,
                                #"train/global_step": self.state.global_step,
                                f"W_top_1_ratio/{key}": top_1_ratio.item(),
                                f"W_top_4_ratio/{key}": top_4_ratio.item(),
                            }
                        )

                        for idx in range(32):
                          wandb.log(
                            {
                              f"W_SV_{key}/idx_{idx+1}": S[idx].item()
                            }
                          )
        self.gradient_accumulation_counter += 1

        return loss.detach() / self.args.gradient_accumulation_steps

    def create_optimizer(self):
        print('Training with Prec-AdamW')
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """


        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None: #True:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            #optimizer_kwargs.update({'optimizer_reg': self.optimizer_reg})
            self.optimizer = AdamWr(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
