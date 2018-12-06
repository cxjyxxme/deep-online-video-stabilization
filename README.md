# Deep Online Video Stabilization
https://arxiv.org/pdf/1802.08091.pdf

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- tensorflow-gpu==1.3.0
- numpy
- ...

## Getting Started
### Installation
```bash
cd deep-online-video-stabilization
cp xxx models/v2_93
```

### Testing
```bash
python3 -u deploy_bundle.py --model-dir ./models/v2_93/ --model-name model-80000 --before-ch 31 --deploy-vis --gpu_memory_fraction 0.9 --output-dir ./output/v2_93/Regular  --test-list /home/ubuntu/Regular/Regular/list.txt --prefix /home/ubuntu/Regular/Regular;
```

### Training
```bash
python -u train_bundle_nobm.py
```
