To use LoRA-One, please use the following script
```
srun python prec_run_exp.py -m ++dataset_name=meta_math +init=gradient ++peft.lora_r=8 +peft=all wandb.name="enter-name-here" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=64 model.learning_rate=2e-5 ++seed=0 ++init.direction="LoRA-One"
```

To use LoRA-One (-), please use the following script
```
srun python run_exp.py -m ++dataset_name=meta_math +init=gradient ++peft.lora_r=8 +peft=all wandb.name="enter-name-here" ++init.weight="stable" peft.use_rslora=True peft.lora_alpha=16 ++init.stable_gamma=64 model.learning_rate=2e-5 ++seed=0 ++init.direction="LoRA-One"
```

The small-scale experiments can be found in
```
Toy_Experiments.ipynb
```
