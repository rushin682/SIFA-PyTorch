## PyTorch Re-implementation: Synergistic Image and Feature Adaptation:<br/> Towards Cross-Modality Domain Adaptation for Medical Image Segmentation

PyTorch implementation of unsupervised cross-modality domain adaptation framework. <br/>
Please refer to [SIFA(TensorFlow)](https://github.com/cchen-cc/SIFA) for the exact version of the network in the AAAI paper (1st Author: Chen, Cheng). <br/>


## Paper
[Synergistic Image and Feature Adaptation: Towards Cross-Modality Domain Adaptation for Medical Image Segmentation](https://arxiv.org/abs/1901.08211)
<br/>
AAAI Conference on Artificial Intelligence, 2019 (oral)
<br/>
<br/>
<!-- ![](figure/framework.png) -->

## Installation
* Still Updating
```
...
```
<!-- git clone https://github.com/cchen-cc/SIFA -->
<!-- cd SIFA -->


## Data Preparation
* Raw data needs to be written into `tfrecord` format to be decoded by `./data_loader.py`. The pre-processed data has been released from [PnP-AdaNet](https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation).
* Put `tfrecord` data of two domains into corresponding folders under `./data` accordingly.
* Run `./create_datalist.py` to generate the datalists containing the path of each data.

## Train
To Be Updated

## Evaluate
To Be updated

<!--
## Citation
If you find the code useful for your research, please cite our paper.
```
@inproceedings{chen2019synergistic,
  author    = {Chen, Cheng and Dou, Qi and Chen, Hao and Qin, Jing and Heng, Pheng-Ann},
  title     = {Synergistic Image and Feature Adaptation:
               Towards Cross-Modality Domain Adaptation for Medical Image Segmentation},
  booktitle = {Proceedings of The Thirty-Third Conference on Artificial Intelligence (AAAI)},
  pages     = {865--872},
  year      = {2019},
}
```
-->

## Acknowledgement
Network Architecture referred from the original [Tensorflow implementation of SIFA](https://github.com/cchen-cc/SIFA).

## Note
* The repository is being updated
* Contact: Rushin Gindra (rhg50@scarletmail.rutgers.edu)
