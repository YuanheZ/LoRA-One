<h1 align="center">
    <p> LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently <br> [ICML2025 (Spotlight)]</p>
</h1>

<h1 align="center"> 
    <img src="./img/lora-one-llama.png" width="300">
</h1>

The Official PyTorch implementation of [**LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently**](https://arxiv.org/abs/2502.01235) [ICML2025 (Spotlight, acceptance rate: ***2.6%***)].

### Content Overview

This paper studies how to improve the performance of Low-Rank Adaption (LoRA) as guided by our theoretical analysis. Our first set of theoretical results show that for linear models: (i) under random initialization, LoRA will align to the certain singular subspace of one-step gradient of full fine-tuning; (ii) preconditioners improve convergence in the high-rank case. These insights motivate us to focus on preconditioned LoRA using a specific spectral initialization strategy for aligning with certain subspaces. For both linear and nonlinear models, we prove that alignment and generalization guarantees can be directly achieved at initialization, and the subsequent linear convergence can be also built. Our analysis leads to the ***LoRA-One*** algorithm (using One-step gradient and preconditioning), a theoretically grounded algorithm that achieves significant empirical improvement over vanilla LoRA and its variants on several benchmarks.

---
### Algorithmic Overview

For each weight matrix, we first compute the full-batch gradient $\nabla_{W} L$ under full fine-tuning and perform SVD on $-\nabla_{W} L$ to get $U$, $\Sigma$, $V$, then we initialize LoRA via
```math
\mathbf{A}_{0}=\frac{1}{\sqrt{\gamma}} U_{[:,:r]} Diag(S[:r])\,,\quad \mathbf{B}_{0}=\frac{1}{\sqrt{\gamma}} Diag(S[:r]) V_{[:,:r]}^\top\,,\quad W_{adapted} = W_{pre}+\frac{\alpha}{\sqrt{r}}\mathbf{A}_{0} \mathbf{B}_{0}\,,
```
which is equivalent to perform one best r-rank full-batch gradient descent under full fine-tuning with learning rate $\frac{\alpha}{\gamma\sqrt{r}}$ at the initialization. The pre-conditioners are modified from [Scale-Adam](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA.git).

---
### Quick Start

Specific config parameters:
```
model:
  bf16: true # set true if needed
  max_length: 1024 # input max length for training
  prec_reg: 1.0e-06 # adjust for pre-conditioners
  saving: false # if true, the model will merge adapters then save after training
init:
  mode: gradient
  direction: LoRA-One
  max_length: 1024 # input max lenght using for computing full-batch gradient, recomment to be consistent with max_length in model
  scale: stable
  stable_gamma: 128 # gamma parameter in the init
  do_subsampling: false # set true to enable using a sub-batch data to initialize
  sub_size: 100000 # specify the size if using sub-batch data
```

To use LoRA-One **without** pre-conditioners, please use the following slurm command
```
srun python run_exp.py -m ++dataset_name=meta_math model.epochs=1 model.eval_epochs=1 ++model.saving=true +init=gradient ++peft.lora_r=8 +peft=all wandb.name="enter-name-here" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=128 model.learning_rate=2e-4 ++seed=9 ++init.direction="LoRA-One"
```

For multi-GPU training, please use the following slurm command (2 GPUs example)
```
CUDA_VISIBLE_DEVICES="0,1" python -m accelerate.commands.launch \
--main_process_port $(shuf -i 10000-60000 -n 1) \
--config_file accelerate_config.yaml \
run_exp.py -m model.epochs=3 model.eval_epochs=3 ++model.saving=false model.real_batch_size=16 ++dataset_name=commonsense_reasoning +init=gradient ++init.direction="LoRA-One" ++init.max_length=256 ++model.max_length=256 ++peft.lora_r=16 +peft=qv wandb.name="enter-name-here" ++init.scale="stable" peft.use_rslora=True peft.lora_alpha=16 ++peft.lora_dropout=0.05 ++init.stable_gamma=128 model.learning_rate=5e-5 ++seed=42
```

The code for LoRA-One **with** pre-conditioners is under revision. We will release once done.

The small-scale (toy) experiments can be found in
```
Toy_Experiments.ipynb
```
---
### Evaluation
To evaluate fine-tuned model on GSM8K by Greedy decoding (recommended for math task), please use the following slurm command:
```
srun python eval_gsm8k.py --model_name="merged_model_path" --wandb_name="enter-name-here"
```
If you want to use top_p sampling instead, please use the following slurm command:
```
srun python eval_gsm8k.py --model_name="merged_model_path" --wandb_name="enter-name-here" --temperature=xxx ---top_p=xxx
```

---

## Citation

If this work is relevant to your research, please cite:

```bibtex
@article{zhang2025one,
  title={One-step full gradient suffices for low-rank fine-tuning, provably and efficiently},
  author={Zhang, Yuanhe and Liu, Fanghui and Chen, Yudong},
  journal={arXiv preprint arXiv:2502.01235},
  year={2025}
}
```
