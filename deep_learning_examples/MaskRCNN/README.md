# Reproduction Notes

Following instructions based on this [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Segmentation/MaskRCNN#quick-start-guide).

Outlined steps and their completion status
- [x] Step 1: Clone the repository.
- [x] Step 2: Build the Mask R-CNN TensorFlow NGC container.
- [x] Step 3: Start an interactive session in the NGC container to run training/inference.

   This command was used instead of the one specified by the NVIDIA documentation.

   ```bash
   docker run --gpus 1 -it --rm --shm-size=10g --ulimit memlock=-1 --ulimit stack=67108864 -v /data/zoey/coco2017_maskrcnn_tf:/data -v /data/zoey/maskrcnn_tf_weights:/weights -v /data/zoey/maskrcnn_tf_results:/results nvidia_mrcnn_tf2
   ```

- [x] Step 4: Download and preprocess the dataset.

  Downloaded to `/data/zoey/coco2017_maskrcnn_tf`

- [x] Step 5: Download the pre-trained ResNet-50 weights.

  Downloaded to `/data/zoey/maskrcnn_tf_weights`

- [x] Step 6: Start training.
  
  This command was used instead of the one specified by the NVIDIA documentation. This modification was made following the [ICSE_2022_artifact](https://github.com/stefanos1316/ICSE_2022_artifact/blob/main/run.sh).

   ```bash
   python main.py train --epochs 1 --steps_per_epoch 1000 --amp --train_batch_size 4
   ```

## Attempts to run outside Docker

### Finding version information from Docker container
Methodology: Within the interactive container session started in Step 3, the version information for the required installations were retrieved with the following commands.
- Python: `python -V`
- TensorFlow: `python -c "import tensorflow as tf;print(tf. __version__)"`
- CUDA Toolkit: `/usr/local/cuda/bin/nvcc --version`
- cuDNN: `cat /usr/include/cudnn_version.h`

Results:
- Python: 3.8.5
- TensorFlow 2.4.0
- CUDA Toolkit 11.2
- cuDNN 8.1.0

Create new conda environment with these versions
1. `conda create --name maskrcnn-tf2 python=3.8.5`
2. `conda activate maskrcnn-tf2`
3. `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
4. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/`
5. `pip install tensorflow=2.4.0`
6. `python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

Steps 3-5 are modified from the [TensorFlow documentation](https://www.tensorflow.org/install/pip).

My first error happens between step 5 and 6. Somehow the Python version changes back to 3.9.12 when I do `python -V`, so when I try to run step 6, I get the error: `ModuleNotFoundError: No module named 'six'`.

Something weird is definitely happening here, but I did not look into why exactly this problem was happening. I just followed what I found here, which told me to deactivate and reactivate the environment: https://stackoverflow.com/questions/36733179/why-conda-cannot-call-correct-python-version-after-activating-the-environment

So, I proceed to deactivate the environment and activate it again, such that python -V shows me 3.8.5.

I then try to run step 6 again, and this time, I get a new error:
```
2023-02-08 11:14:33.668530: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcusolver.so.10'; dlerror: libcusolver.so.10: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/cuda-11.2/lib64:/usr/local/cuda-11.0/lib64:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/:/home/zoey/miniconda3/envs/maskrcnn-tf2/lib/
2023-02-08 11:14:33.669389: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1757] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
[]
```

And this seems to be the main problem where the GPU is not being successfully registered by TensorFlow, which is where I am stuck at currently. I think there might be a problem with my paths in Step 4 that is causing the missing file.
