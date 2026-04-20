# [ICLR 2026] Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-Promoting Fine-Tuning

This repository contains the official code of our ICLR 2026 paper:

> **Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-Promoting Fine-Tuning**
> 
> Youngwoo Cho\*, Seunghoon Yi\*, Wooil Yang, Sungmo Kang, Young-Woo Son, Jaegul Choo, Joonseok Lee, Soo Kyung Kim, Hongkee Yoon
> *International Conference on Learning Representations (ICLR), 2026*
> (\*Equal contribution)

📄 **OpenReview**: https://openreview.net/forum?id=moBqB1CUym

> ⚠️ Only the core algorithm (`core.py`) is released at the moment. Environment setup and full training scripts will be updated soon.

## Core Algorithm
`core.py` provides `EquiSparseDeltaSTR`, a wrapper that replaces `e3nn`'s `Linear` and `FullyConnectedTensorProduct` layers with their sparse-delta counterparts. The original pretrained weights are frozen, and a learnable delta weight is trained with a soft-threshold operator to induce sparsity.

The following snippet illustrates the conceptual usage of `EquiSparseDeltaSTR`. It is not a runnable script — see the upcoming training code for the full pipeline.

```python
from core import EquiSparseDeltaSTR

model = EquiSparseDeltaSTR(e3nn_model, init_threshold = 1e-4, per_instruction = False)
 
# separate parameter groups: delta weights vs. threshold scores
delta_params = list()
score_params = list()

for name, param in model.named_parameters():

    if not param.requires_grad:

        continue

    (score_params if 'score' in name else delta_params).append(param)

# sparsity is induced by weight decay on scores
optimizer = torch.optim.AdamW([
    {'params': delta_params, 'lr': 1e-3, 'weight_decay': 0.0},
    {'params': score_params, 'lr': 1e-3, 'weight_decay': 1e-4}])
 
# training loop
for batch in loader:

    loss = compute_loss(model(batch), batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 
# for deployment: commit pruned delta into the original weights
model.prune()
model.merge()
```

## Environment
```shell
Coming soon.
```

## How to run?
```shell
Coming soon.
```

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{
cho2026sparsitypromoting,
title={Robust and Interpretable Adaptation of Equivariant Materials Foundation Models via Sparsity-promoting Fine-tuning},
author={Youngwoo Cho and Seunghoon Yi and Wooil Yang and Sungmo Kang and Young-Woo Son and Jaegul Choo and Joonseok Lee and Soo Kyung Kim and Hongkee Yoon},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=moBqB1CUym}
}
```
