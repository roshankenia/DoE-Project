# Faster-RCNN pytorch
This is a pytorch implementation of detecting digits on pebbles with faster-rcnn, which can either run with jupyter notebook or with cmd

## Setup

### Prerequisites

- CUDA  10.1
- python  3.7
- pytorch  1.12.1
- torchvision  0.13.1
- opencv-python  4.6.0.66

Note: This implementation is tested on these prerequisites. Other versions of the prerequisites may also work, but I do not try.


### Getting Started

- Train and test the model on our own dataset

```bash
detection_20220901_600.ipynb (if you want to run with jupyter notebook)
```
otherwise you can directly run

```bash
python detection_20220901_600.py
```

or equivalently
./run.sh
or
./run_nohop.sh (you do not need to worry about the network disconnection if you run this)

Note: you need to put the dataset under the same folder with the codes, and change the root of the dataset in the 123th line of "detection_20220901_600.py"

