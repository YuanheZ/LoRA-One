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
which is equivalent to perform one best r-rank full-batch gradient descent under full fine-tuning at the initialization. The pre-conditioners are modified from [Scale-Adam](https://github.com/pilancilab/Riemannian_Preconditioned_LoRA.git).

---

To use LoRA-One without pre-conditioners, please use the following script
```
srun python run_exp.py -m ++dataset_name=meta_math +init=gradient ++peft.lora_r=8 +peft=all wandb.name="enter-name-here" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=64 model.learning_rate=2e-5 ++seed=0 ++init.direction="LoRA-One"
```

The small-scale experiments can be found in
```
Toy_Experiments.ipynb
```

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
