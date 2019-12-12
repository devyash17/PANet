# Path Aggregation Network for Instance Segmentation

by [Shu Liu](http://shuliu.me), Lu Qi, Haifang Qin, [Jianping Shi](https://shijianping.me/), [Jiaya Jia](http://jiaya.me/).


## Installation

For environment requirements, data preparation and compilation, please refer to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

Note: Changes need to be made in [dataset_catalog.py](lib/datasets/dataset_catalog.py) and [dummy_datasets.py](lib/datasets/dummy_datasets.py) according to the dataset you are going to use.

WARNING: pytorch 0.4.1 is broken, see https://github.com/pytorch/pytorch/issues/8483. Use pytorch 0.4.0

## Usage

To train PANet, simply use corresponding config files. For example, to train PANet on COCO:

```shell
python3 tools/train_net_step.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
```

To visualize the predictions, use:
```shell
python3 tools/infer_simple.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --load_ckpt {path/to/your/checkpoint} --image_dir {path/to/the/images} --output_dir {path/to/the/output/directory}
```

## Evaluation (for bbox only)

To evaluate the model, apply the below given instructions in the order shown:
* Extract all the predictions using [extract_pred.py](tools/extract_pred.py). Change `pred_path` and `no_of_labels` as required and run the command:
```shell
python3 tools/extract_pred.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --load_ckpt {path/to/your/checkpoint} --image_dir {path/to/the/images}
```
  It will store the predictions in the `pred_path` directory in separate json files labelwise.

* See [this](https://github.com/devyash17/mAP) for further instructions.


## Main Results

| Labels   | IoU = 0.5   | IoU = 0.55   | IoU = 0.6  | IoU = 0.65  | IoU = 0.7  | IoU = 0.75  | IoU = 0.8  | IoU = 0.85  |  IoU = 0.9  |  IoU = 0.95
 | :------------: | --------------- | -------------- | ------------- | --------------- | ---------------- | ----------------- | ------------- | --------------- | -------------- | ------------- |
 signage | 82.34% | 80.77% | 78.36%  | 73.95% | 67.53%  | 58.08% | 42.90% | 24.08% | 6.46% | 0.11% |
 traffic_sign | 84.40% | 83.18% | 80.63% | 75.60% | 68.21% |  55.21% | 37.33% | 17.38% | 3.46% | 0.06% |
 traffic_light | 65.65% | 60.43% | 53.56% | 43.65%  | 32.22% |  20.14% | 9.80% | 2.82% | 0.41% | 0.01% |

### P-R curves
1. **signage**
![image](https://user-images.githubusercontent.com/41137582/70707624-8eb94980-1cfe-11ea-8d18-29a2d2eabbc9.png)
2. **traffic_light**
![image](https://user-images.githubusercontent.com/41137582/70707700-c7f1b980-1cfe-11ea-9725-0bcccbeb7113.png)
3. **traffic_sign**
![image](https://user-images.githubusercontent.com/41137582/70707715-d17b2180-1cfe-11ea-9d3a-e17b71f7f8a7.png)

### Final mAP (for bbox)
![image](https://user-images.githubusercontent.com/41137582/70708036-975e4f80-1cff-11ea-9b95-d91c59f8b767.png)
