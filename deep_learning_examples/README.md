# Task updates

| Model (TensorFlow2) | Category                    | Status                                                   | Time to Train | Time to Infer |
|---------------------|-----------------------------|----------------------------------------------------------|---------------|---------------|
| Mask R-CNN          | Computer Vision             | Finished without issues                                  |               |               |
| EfficientNet v1-B0  | Computer Vision             | Downloading ImageNet dataset                             |               |               |
| EfficientNet v1-B4  | Computer Vision             | Downloading ImageNet dataset                             |               |               |
| EfficientNet v2-S   | Computer Vision             | Downloading ImageNet dataset                             |               |               |
| U-Net Med           | Computer Vision             | Dataset download link does not work                      |               |               |
| DLRM                | Recommender Systems         |                                                          |               |               |
| Wide&Deep           | Recommender Systems         | Dataset download from Kaggle API returns 404 - Not Found |               |               |
| SIM                 | Recommender Systems         | Preprocessing of dataset results in CUDA out of memory   |               |               |
| ELECTRA             | Natural Language Processing |                                                          |               |               |
| BERT                | Natural Language Processing |                                                          |               |               |

## ICSE_22 Tasks

| Task          | Original Docker Container Version | New Docker Container Version | Status                                                                  |
|---------------|-----------------------------------|------------------------------|-------------------------------------------------------------------------|
| Mask RCNN     | tensorflow:21.02-tf2-py3          | -                            | Training and inference OK, passed to Saurabh to run client-side program |
| ResNet-50     | tensorflow:20.12-tf1-py3          |                              | Error with data, tfrecord corrupted?                                    |
| GNMT          | tensorflow:20.06-tf1-py3          | tensorflow:20.12-tf1-py3     | Has been training for 2.5h, had to reduce batch size                    |
| NCF           | tensorflow:20.07-tf1-py3          | TBC                          | Asked Saurabh to run if he has time                                     |
| SSD           | tensorflow:20.06-tf1-py3          | TBC                          | Asked Saurabh to run if he has time                                     |
| Transfomer-XL |                                   |                              |                                                                         |

- [ ] SSD
- [ ] GNMT
- [ ] Transformer-XL
- [ ] NCF
- [ ] ResNet-50
- [x] Mask RCNN

GNMT: Original repository uses tensorflow:20.06-tf1-py3, which is NOT compatible with our GPU.
I changed it to tensorflow:20.12-tf1-py3, seems okay so far? (As of 24 Feb 9:30pm)

ResNet-50: Original repository uses tensorflow:20.12-tf1-py3, which seems to be compatible with our GPU!
Problem with data. Corrupted tfrecord

SSD: Asked Saurabh to try if he has time. Original repository uses tensorflow:20.06-tf1-py3, which should not work with our GPU (to be verified). Change to tensorflow:20.12-tf1-py3?

NCF: Original repository uses tensorflow:20.07-tf1-py3, not sure if it works with our GPU? (VERIFY)

## New Repo/Project Search Tasks

| Repo Name | Repo Link | Forks | Stars | Issues | Status |
|-----------|-----------|-------|-------|--------|--------|
|           |           |       |       |        |        |
|           |           |       |       |        |        |


