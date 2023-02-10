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
