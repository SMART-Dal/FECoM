# Reproduction Notes for EfficientDet

Following instructions based on this [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Detection/Efficientdet#quick-start-guide).

Outlined steps and their completion status
- [x] Step 1: Clone the repository.
- [x] Step 2: Download and preprocess the dataset.
  Location: `/data/zoey/coco`
- [x] Step 3: Build the EfficientDet TensorFlow NGC container.
- [x] Step 4: Start an interactive session in the NGC container to run training/inference
  In this step, it was mentioned that an EfficientNet checkpoint needs to be present. I don't think they specified which one but I used the [EfficientNet-V1-B4 checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/models/efficientnet_tf2_ckpt_v1_b4_amp).
  Its location is in /data/zoey
  DATA=/data/zoey/coco BACKBONE_CKPT=/data/zoey/model.ckpt-0500.data-00000-of-00001 bash scripts/docker/interactive.sh
- [ ] Step 5: Start training
```
bash ./scripts/D0/convergence-AMP-8xV100-32G.sh
```
This leads to an error, as the bash script seems to be for 8 GPUs, which we do not have.


## Attempts to run outside Docker

### Finding version information from Docker container
Methodology: Within the interactive container session started in Step 3, the version information for the required installations were retrieved with the following commands.
- Python: `python -V`
- TensorFlow: `python -c "import tensorflow as tf;print(tf. __version__)"`
- CUDA Toolkit: `/usr/local/cuda/bin/nvcc --version`
- cuDNN: `cat /usr/include/cudnn_version.h`

Results:
- Python: 3.8.10
- TensorFlow 2.8.0
- CUDA Toolkit 11.6
- cuDNN 8.3.3

1. conda create --name efficientdet-tf2 python=3.8.10
2. conda activate efficientdet-tf2
3. conda install -c conda-forge cudatoolkit=11.6 cudnn=8.3.3
4. 
   PackagesNotFoundError: The following packages are not available from current channels:

  - cudnn=8.3.3

Try this? https://developer.nvidia.com/cudnn
