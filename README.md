<h1 align="center">
    <p> LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently <br> [ICML2025 (Spotlight)]</p>
</h1>

<h1 align="center"> 
    <img src="./img/lora-one-llama.png" width="300">
</h1>

The Official PyTorch implementation of [**LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently**](https://arxiv.org/abs/2502.01235) [ICML2025 (Spotlight, acceptance rate: ***2.6%***)].

This paper studies how to improve the performance of Low-Rank Adaption (LoRA) as guided by our theoretical analysis. Our first set of theoretical results show that for random initialization and linear models, \textit{i)} LoRA will align to the certain singular subspace of one-step gradient of full fine-tuning; \textit{ii)} preconditioners improve convergence in the high-rank case. These insights motivate us to focus on preconditioned LoRA using a specific spectral initialization strategy for aligning with certain subspaces. For both linear and nonlinear models, we prove that alignment and generalization guarantees can be directly achieved at initialization, and the subsequent linear convergence can be also built. Our analysis leads to the \emph{LoRA-One} algorithm (using \emph{One}-step gradient and preconditioning), a theoretically grounded algorithm that achieves significant empirical improvement over vanilla LoRA and its variants on several benchmarks.

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
