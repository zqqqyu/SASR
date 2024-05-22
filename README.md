# Non-Local Degradation Modeling for Spatially Adaptive Single Image Super-resolution

## For training and testing
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

1.2 Combine the HR images from these two datasets in `your_data_path/DF2K/HR` to build the DF2K dataset. 
refer to [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR)

For training using `main.sh`:
```bash
CUDA_VISIBLE_DEVICES=0 sh main.sh
```
For testing using `test.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 sh test.sh
```
