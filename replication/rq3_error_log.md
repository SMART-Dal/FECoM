# RQ3 Issue Log
This file provides a detailed list of issues faced, arranged by categories and subcategories.

## Patching
### Patch Generation

#### Correctness of Patches

#### Patch Granularity

## Energy Measurement
### Instrumentation Challenges

#### Code instrumentation

#### Noise in measurement

### Hardware Variability

#### Energy Measurement Accuracy

#### Calibration Issues

#### GPU Usage

### Granularity of Energy Attribution

#### Precision Level

#### Precision Limits

## Framework Design and Implementation
### Framework Extensibilty

## Environment
### Hardware incompatibility

### GPU challenges

#### Memory Management

#### Container Issues
This issue was encountered while experimenting with the NVIDIA [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master) dataset.
The examples in the dataset are provided in Docker containers.
However, for some examples, the specified version of the TensorFlow container is not compatible with the GPU, resulting in the following error:
```
  WARNING: Detected NVIDIA NVIDIA GeForce RTX 3070 Ti GPU, which is not yet supported in this version of the container
  ERROR: No supported GPU(s) detected to run this container
  ```
Following exact specified instructions to run projects on Docker container is sometimes not possible, due to incompatibility with GPU (see error log).
But modifying (choosing a newer version) the tensorflow image version can cause runtime errors too

## Data Handling
### Large Datasets

### Serialisation
