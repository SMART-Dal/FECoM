# Reproduction Notes for GNMT

Following instructions based on this [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Translation/GNMT#quick-start-guide).

Outlined steps and their completion status
- [x] Step 1: Clone the repository.
- [x] Step 2: Build the GNMT v2 TensorFlow container.

  Change the Dockerfile container version from 20.06 to 20.12. If not, the following error will occur:
  ```
  WARNING: Detected NVIDIA NVIDIA GeForce RTX 3070 Ti GPU, which is not yet supported in this version of the container
  ERROR: No supported GPU(s) detected to run this container
  ```
- [x] Step 3: Start an interactive session in the NGC container to run training/inference.
- [x] Step 4: Download and preprocess the dataset.

  IMPORTANT: The download links in the bash script are outdated. Change all download links from http to https!
- [ ] Step 5: Start training.
  
  The script ran (modified from ICSE22 repository, changed batch size from 128 to 32 to prevent out-of-memory issues):
  ```
  nvidia-docker run -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $PWD:/workspace/gnmt/ -v /data/zoey/:/workspace/gnmt/data/ gnmt_tf python nmt.py --output_dir=results --batch_size=32 --learning_rate=2e-3
  ```

  Experiment started at 7:30pm, still running at 9:24pm! In the ICSE paper, GNMT TensorFlow ran for around 2h.
  Our experiment may take longer due to the reduced batch size.




At least for GNMT and ResNet-50, the GPU seems to be working