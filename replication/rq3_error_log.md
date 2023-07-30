# RQ3 Issue Log
This file provides a detailed list of issues faced.
The list of issues faced are grouped into 4 categories:
1. [Energy Measurement](#energy-measurement)
2. [Patching](#patching)
3. [Execution Environment](#execution-environment)
4. [Framework Design and Implementation](#framework-design-and-implementation)

Within each category, the issues are further grouped into subcategories, as well as sub-subcategories, where applicable.
## Energy Measurement   

| Subcategory                       | Codes                       | Details |
|-----------------------------------|-----------------------------|---------|
| Instrumentation Challenges        | Code instrumentation        |         |
| Instrumentation Challenges        | Noise in measurement        |         |
| Hardware Variability              | Energy Measurement Accuracy |         |
| Hardware Variability              | Calibration Issues          |         |
| Hardware Variability              | GPU Usage                   |         |
| Granularity of energy attribution | Precision limits            |         |
| Granularity of energy attribution | Precision overhead balance  |         |

## Patching

| Subcategory      | Codes                  | Details |
|------------------|------------------------|---------|
| Patch Generation | Correctness of Patches |         |
| Patch Generation | Patch Granularity      |         |

## Execution Environment

| Subcategory              | Codes             | Details                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|--------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Hardware incompatibility | -                 |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| GPU challenges           | Memory Management | A CUDA runtime error stating that the GPU has ran out of memory occurs when a process cannot be allocated sufficient memory:<pre>std::bad_alloc: CUDA error at: ../include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory</pre>                                                                                                                                                                                                                                                                                                    |
| GPU challenges           | Container Issues  | This issue was encountered while experimenting with the NVIDIA [DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples/tree/master) dataset. The examples in the dataset are provided in Docker containers. However, for some examples, the specified version of the TensorFlow container is not compatible with the GPU, resulting in the following error: <pre>WARNING: Detected NVIDIA NVIDIA GeForce RTX 3070 Ti GPU, which is not yet supported in this version of the container<br>ERROR: No supported GPU(s) detected to run this container</pre> |

## Framework Design and Implementation
### Framework Extensibility
