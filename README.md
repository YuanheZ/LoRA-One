<h1 align="center">
    <p> LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently <br> [ICML2025 (Spotlight)]</p>
</h1>

<h1 align="center"> 
    <img src="./img/lora-one-llama.png" width="300">
</h1>

The Official PyTorch implementation of [**LoRA-One: one-step full gradient suffices for low-rank fine-tuning, provably and efficiently**](https://arxiv.org/abs/2502.01235) [ICML2025 (Spotlight, acceptance rate: ***2.6%***)].

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
