## Attention Contact Planning (wip)

Implementation of <a href="https://leggedrobotics.github.io/ACP/">Humanoid Mobility via Attention-based Contacts</a> (<a href="https://arxiv.org/abs/2610.31608">paper</a>), out of ETH Zürich

## Install

```bash
pip install attention-contact-planning
```

## Usage

The proposed cross attention, where body queries attend over terrain points, with a learned no-contact slot at index 0

```python
import torch
from attention_contact_planning import CrossAttention

cross_attn = CrossAttention(
    dim = 96,
    heads = 32,
    dim_head = 3,
    point_dim = 3
)

body_queries = torch.randn(2, 9, 96)     # b i d
terrain_points = torch.randn(2, 187, 3)  # b j 3
point_mask = torch.ones(2, 187, dtype = torch.bool)

out = cross_attn(
    body_queries,
    terrain_points,
    point_mask = point_mask
)

out.contact_probs # (2, 32, 9, 188), column 0 is no-contact
```

Train the contact distribution against soft targets straight from the attention logits

```python
from attention_contact_planning import contact_probs_loss

soft_target_probs = torch.randn(2, 9, 187).softmax(dim = -1) # b i j

loss = contact_probs_loss(
    out.attn_logits,
    soft_target_probs,
    point_mask = point_mask
)
```

## Citations

```bibtex
@article{acp2026humanoid,
    title   = {Humanoid Mobility via Attention-based Contacts},
    author  = {He, Junzhe and Wen, Kehan and Zhang, Chong and Schwarke, Clemens and
               Zhang, Zewei and Qu, Kaixian and Hu, Siyuan and M{\"u}ller, Matthias and
               Koltun, Vladlen and Hutter, Marco},
    journal = {arXiv preprint arXiv:2610.31608},
    year    = {2026}
}
```

```bibtex
@article{shazeer2020talkingheads,
    title   = {Talking-Heads Attention},
    author  = {Shazeer, Noam and Lan, Zhenzhong and Cheng, Youlong and Ding, Nan and Hou, Le},
    journal = {arXiv preprint arXiv:2003.02436},
    year    = {2020}
}
```

```bibtex
@article{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Jaegle, Andrew and Gimeno, Felix and Brock, Andrew and Zisserman, Andrew and
               Vinyals, Oriol and Carreira, Jo{\~a}o},
    journal = {arXiv preprint arXiv:2103.03206},
    year    = {2021}
}
```
