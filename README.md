# Deep Online Video Stabilization with Multi-Grid Warping Transformation Learning
https://ieeexplore.ieee.org/document/8554287

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN
- tensorflow-gpu==1.3.0
- numpy
- ...

## Getting Started
### Installation
Download data.zip at https://cg.cs.tsinghua.edu.cn/people/~miao/stabnet/data.zip.

This dataset does not contain flow information(set to 0). If you need to use data containing flow information, you can use the TVL1 algorithm to generate it.
```bash
unzip data.zip
mv data/models deep-online-video-stabilization/
mv data/datas deep-online-video-stabilization/
mv data/data deep-online-video-stabilization/
cd deep-online-video-stabilization-deploy
mkdir output
```

### Testing
```bash
python3 -u deploy_bundle.py --model-dir ./models/v2_93/ --model-name model-80000 --before-ch 31 --deploy-vis --gpu_memory_fraction 0.9 --output-dir ./output/v2_93/Regular  --test-list /home/ubuntu/Regular/Regular/list.txt --prefix /home/ubuntu/Regular/Regular;
```

### Training
```bash
python -u train_bundle_nobm.py
```
### Dataset
DeepStab dataset (7.9GB)
    http://cg.cs.tsinghua.edu.cn/download/DeepStab.zip

## Citation

    If you find this useful for your research, please cite the following paper.

    ```
    @ARTICLE{StabNet, 
        author={M. Wang and G. Yang and J. Lin and S. Zhang and A. Shamir and S. Lu and S. Hu}, 
        journal={IEEE Transactions on Image Processing}, 
        title={Deep Online Video Stabilization with Multi-Grid Warping Transformation Learning}, 
        year={2018}, 
        volume={}, 
        number={}, 
        pages={1-1}, 
    }
    ```

