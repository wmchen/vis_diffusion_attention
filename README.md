## Installation

Create `conda` environment:

```bash
conda create -n flux_attention python=3.10
conda activate pytorch
```

Install pytorch:

```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Install other packages:

```bash
pip install -r requirements.txt
```
