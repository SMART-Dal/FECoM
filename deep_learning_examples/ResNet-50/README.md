# Reproduction Notes for ResNet-50

Following instructions based on this [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Classification/ConvNets/resnet50v1.5#quick-start-guide).

Outlined steps and their completion status
- [x] Step 1: Clone the repository.
- [x] Step 2: Download and preprocess the dataset.
The ResNet50 v1.5 script operates on ImageNet 1k.
[This link](http://image-net.org/download-images) to download the dataset was provided by NVIDIA.
It requires login and I couldn't really navigate to the intended download location.
Thus, I downloaded from Kaggle (in TFrecord format, essentially skipping the data preprocessing step of the guide).
   ```
   kaggle datasets download -d hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0
   unzip imagenet-1k-tfrecords-ilsvrc2012-part-0.zip
   kaggle datasets download -d hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1
   unzip imagenet-1k-tfrecords-ilsvrc2012-part-1.zip
   mv train-* train/
   ```
  Location of data: `/data/zoey/imagenet_1k`
- [x] Step 3: Build the ResNet-50 v1.5 TensorFlow NGC container.
- [x] Step 4: NOT IN GUIDE - Modify the `DGX1_RN50_AMP_90E.sh` file by replacing it with the [file](https://github.com/stefanos1316/ICSE_2022_artifact/blob/main/configs/tensorflow_rn50_DGX1_RN50_AMP_90E.sh) specified in the ICSE22 repo.
- [ ] Step 5: Start an interactive session in the NGC container to run training/inference. Modified according to the ICSE22 repo:
  ```shell
  nvidia-docker run --rm -it -v /data/zoey/imagenet_1k/train:/data/tfrecords --ipc=host nvidia_rn50 bash resnet50v1.5/training/DGX1_RN50_AMP_90E.sh
  ```

  Got error: DataLossError: corrupted record at xxx

  Things to try:
  - Download raw data again and do the preprocessing step myself (possibly time-consuming)
  - Run script to check for the corrupted files and drop them? Refer to: https://gist.github.com/ed-alertedh/9f49bfc6216585f520c7c7723d20d951




