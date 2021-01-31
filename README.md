# This rep is pytorch re-implementation of [AdvancedEast](https://github.com/huoyijie/AdvancedEAST), and the project structure reference from [DBNet.pytorch](https://github.com/WenmuZhou/DBNet.pytorch)


## Data Preparation

Training data: prepare a text `train.txt` in the following format, use '\t' as a separator
```
./datasets/train/img/001.jpg	./datasets/train/gt/001.txt
```

Validation data: prepare a text `test.txt` in the following format, use '\t' as a separator
```
./datasets/test/img/001.jpg	./datasets/test/gt/001.txt
```
- Store images in the `img` folder
- Store groundtruth in the `gt` folder

The groundtruth can be `.txt` files, with the following format:
```
x1, y1, x2, y2, x3, y3, x4, y4, annotation
```
## train

1. config network, data_path ...  in config/advanced_east.yaml
2. python train.py

## val

python eval.py

## predict

python predict.py

## examples

激活图
![examples](examples/activation.png)
检测图
![examples](examples/prediction.png)


## TODO

* data_augment


### reference
1. https://github.com/huoyijie/AdvancedEAST
2. https://github.com/WenmuZhou/DBNet.pytorch