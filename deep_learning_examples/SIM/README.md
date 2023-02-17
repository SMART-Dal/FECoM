- [x] Step 1: Clone the repository.
- [ ] Step 2: Build the SIM Tensorflow2 container.
- [ ] Step 3: Start an interactive session in the NGC container to run preprocessing, training, or inference
```
docker run --runtime=nvidia -it --rm --ipc=host --security-opt seccomp=unconfined -v /data/zoey/amazon_books_2014:/data/amazon_books_2014 -v /data/zoey/sim_parquet:/data/sim_parquet sim_tf2 bash
```
- [ ] Step 4: Download Amazon Books dataset
  In location: /data/zoey/amazon_books_2014
python preprocessing/sim_preprocessing.py \
 --amazon_dataset_path /data/amazon_books_2014 \
 --output_path /data/sim_parquet

Problem: Encountered MemoryError: std::bad_alloc: CUDA error at: ../include/rmm/mr/device/cuda_memory_resource.hpp:70: cudaErrorMemoryAllocation out of memory

Possible reason: Even for preprocessing, it makes use of loading data to GPU. The reviews_Books.json file is 19GB which is larger than GPU memory.