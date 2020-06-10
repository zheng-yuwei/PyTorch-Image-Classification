# 基于PyTorch的分类网络库

实现的分类网络包括：

- [x] PyTorch自带的网络：resnet, shufflenet, densenet, mobilenet, mnasnet等；
- [x] MobileNet v3；
- [x] EfficientNet系列；
- [x] ResNeSt系列（实验达到各项STOA，但是论文被SR了）;

---

## 包含特性

- [x] 支持多种功能（`application/`）：训练、测试、转JIT部署、模型蒸馏、可视化；
- [x] 数据增强（`dataloader/enhancement`)：AutoAugment，自定义Augment(MyAugment)，mixup数据增强，多尺度训练数据增强；
- [x] 库中包含多种优化器（`optim`）：目前使用的是Adam，同时推荐RAdam；
- [x] 不同损失指标的实现（`criterions`）：OHM、GHM、weighted loss等；

---

## 文件结构说明

- `applications`: 包括`test.py, train.py, convert.py`等应用，提供给`main.py`调用；
- `checkpoints`: 训练好的模型文件保存目录（当前可能不存在）；
- `criterions`: 自定义损失函数；
- `data`: 训练/测试/验证/预测等数据集存放的路径；
- `dataloader`: 数据加载、数据增强、数据预处理（默认采用ImageNet方式）；
- `demos`: 模型使用的demo，目前`classifier.py`显示如何调用`jit`格式模型进行预测；
- `logs`: 训练过程中TensorBoard日志存放的文件（当前可能不存在）；
- `models`: 自定义的模型结构；
- `optim`: 一些前沿的优化器，PyTorch官方还未实现；
- `pretrained`: 预训练模型文件；
- `utils`: 工具脚本：混淆矩阵、图片数据校验、模型结构打印、日志等；
- `config.py`: 配置文件；
- `main.py`: 总入口；
- `requirements.txt`: 工程依赖包列表；

---

## 使用说明

### 数据准备

在文件夹`data`下放数据，分成三个文件夹: `train/test/val`，对应 训练/测试/验证 数据文件夹；
每个子文件夹下，依据分类类别每个类别建立一个对应的文件夹，放置该类别的图片。

数据准备完毕后，使用`utils/check_images.py`脚本，检查图像数据的有效性，防止在训练过程中遇到无效图片中止训练。

最终大概结构为：
```
- data
  - train
    - class_0
      - 0.jpg
      - 1.jpg
      - ...
    - class_1
      - ...
    - ..
  - test
    - ...
  - val
    - ...
- dataloader
- ...
```

### 部分重要配置参数说明

针对`config.py`里的部分重要参数说明如下：

- `--data`: 数据集根目录，下面包含`train`, `test`, `val`三个目录的数据集，默认当前文件夹下`data/`目录；
- `--image_size`: 输入应该为两个整数值，预训练模型的输入时正方形的，也就是[224, 224]之类的；
实际可以根据自己需要更改，数据预处理时，会将图像 等比例resize然后再padding（默认用0 padding）到 指定的输入尺寸。
- `--num_classes`: 分类模型的预测类别数；
- `-b`: 设置batch size大小，默认为256，可根据GPU显存设置；
- `-j`: 设置数据加载的进程数，默认为8，可根据CPU使用量设置；
- `--criterion`: 损失函数，一种使用PyTorch自带的softmax损失函数，一种使用我自定义的sigmoid损失函数；
sigmoid损失函数则是将多分类问题转化为多标签二分类问题，同时我增加了几个如GHM自定义的sigmoid损失函数，
可通过`--weighted_loss --ghm_loss --threshold_loss --ohm_loss`指定是否启动；
- `--lr`: 初始学习率，`main.py`里我默认使用Adam优化器；目前学习率的scheduler我使用的是`LambdaLR`接口，自定义函数规则如下，
详细可参考`main.py`的`adjust_learning_rate(epoch, args)`函数：
```
~ warmup: 0.1
~ warmup + int([1.5 * (epochs - warmup)]/4.0): 1, 
~ warmup + int([2.5 * (epochs - warmup)]/4.0): 0.1
~ warmup + int([3.5 * (epochs - warmup)]/4.0) 0.01
~ epochs: 0.001
```
- `--warmup`: warmup的迭代次数，训练前warmup个epoch会将 初始学习率*0.1 作为warmup期间的学习率；
- `--epochs`: 训练的总迭代次数；
- `--aug`: 是否使用数据增强，目前默认使用的是我自定义的数据增强方式：`dataloader/my_augment.py`；
- `--distill`: 模型蒸馏（需要教师模型输出的概率文件)，默认 False，
使用该模式训练前，需要先启用`--knowledge train --resume 好的模型`对训练集进行测试，生成概率文件作为教师模型的概率；
概率文件形式为`data`路径下`distill*.txt`模式的文件，有多个文件会都使用，取均值作为教师模型的概率输出指导接下来训练的学生模型；
- `--mixup`: 数据增强mixup，默认 False；
- `--multi_scale`: 多尺度训练，默认 False；
- `--resume`: 权重文件路径，模型文件将被加载以进行模型初始化，`--jit`和`--evaluation`时需要指定；
- `--jit`: 将模型转为JIT格式，利于部署；
- `--evaluation`: 在测试集上进行模型评估；
- `--knowledge`: 指定数据集，使用教师模型（配合resume选型指定）对该数据集进行预测，获取概率文件（知识)，
生成的概率文件路径为`data/distill.txt`，同时生成原始概率`data/label.txt`;
- `--visual_data`: 对指定数据集运行测试，并进行可视化；
- `--visual_method`: 可视化方法，包含`cam`, `grad-cam`, `grad-camm++`三种；
- `--make_curriculum`: 制作课程学习的课程文件；
- `--curriculum_thresholds`: 不同课程中样本的阈值；
- `--curriculum_weights`: 不同课程中样本的损失函数权重；
- `--curriculum_learning`: 进行课程学习，从`data/curriculum.txt`中读取样本权重数据，训练时给对应样本的损失函数加权；

BTW，在`models/efficientnet/model.py`中增加了`sample-free`的思想，目前代码注释掉了，若需要可以借鉴使用。
`sample-free`主要是我使用bce进行多标签二分类时，我希望任务偏好某些类别，所以在初始某些类别的bias上设置一个较大的数，提高初始概率。
（具体计算公式可参考原论文 Is Sampling Heuristics Necessary in Training Deep Object Detectors）

参数的详细说明可查看`config.py`文件。

---

## 快速使用

可参考对应的`z_task_shell/*.sh`文件

### 模型部署demo

训练好模型后，想用该模型对图像数据进行预测，可使用`demos`目录下的脚本`classifier.py`：

```shell
cd demos
python classifier.py
```

---

## 模型对比

| 模型                 | Params(M) | FLOPs(M) | mac-cpu前向（ms）| 准确率%（自己数据） |
| ------------------- |:---------:|:--------:|:--------------:|:----------------:|
| resnet18            | 11.18     | 1820.16  | 65             | -                |
| resnet50            | 23.52     | 4109.68  | 160            | -                |
| shufflenet_v2_x1_0  |  1.26     |  148.00  | 25             | -                |
| shufflenet_v2_x2_0  |  5.36     |  588.43  | 58             | -                |
| mobilenet_v2        |  2.23     |  306.18  | 52             | -                |
| mobilenetv3_small   |  1.52     |   57.38  | 20             | -                |
| mobilenetv3_large   |  4.21     |  222.04  | 50             | -                |
| efficientnet_b0     |  4.00     |  405.31  | 135            | -                |
| efficientnet_b4     | 18.04     | 4600.46  | 320            | -                |

* 输入图像尺寸为[224, 224]；
* 类别数为 6分类；
* efficientnet的params和flops是预估的。

### 训练策略

| 损失函数              | 准确率%（自己实际数据） |
| ------------------- |:-------------------:|
| softmax             | 89.2                |
| bce loss            | 89.0                |
| threshold bce       | 88.95               |
| ghm bce             | 89.3                |
| weighted bce        | 89.1                |
| ohm bce             | 89.1                |

* EfficientNet b0，data augmentation，Adam step LR with warm-up
* 单看整体指标可能不太好，以上某些类别的指标

`Threshold Loss` 相比 `BCELoss` 指标相当，而`OHM Loss`略微上升，表明调整不同样本在优化过程的权重，可得到更好的训练结果。

### 数据增强对比

| 增强方式                     | 准确率%（自己实际数据） |
| -------------------------- |:-------------------:|
| baseline(bce+aug)          | 89.00               |
| -pretrained                | 85.97               |
| -aug                       | 87.76               |
| +multi-scale               | 89.02               |
| +mixup                     | 88.63               |
| +distill from b0+ghm       | 88.55               |

### 本库的一些Tips

- 本库多GPU训练时，使用DDP进行分布式训练（无论是否单机）；
若是多机，每台服务器都会保存日志和模型，由于同步模型初始参数、梯度、BN，所以模型应该是一致的；日志可能略有差异；
- 在进行模型评估时，会使用all_together将不同GPU上的结果汇总，所以指标上应该是一致的；
- Weighted Loss、GHM-C Loss效果有所提升好；ThresholdLoss需要调参；
- multi-scale几乎无效果，且warm-up后准确率波动很大，可能因为学习率没调；
- distill：使用GHM-C的b0作为teacher效果变差（和原label平均，效果差更多了），但若使用更好的teacher，则提升明显；
- 我看github上mix-up有两种：image和target进行mix、image和loss进行mix，两种都变差，第一种效果差得很；

---

## Reference

[d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)

[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)

[zhanghang1989/ResNeSt](https://github.com/zhanghang1989/ResNeSt)

[yizt/Grad-CAM.pytorch](https://github.com/yizt/Grad-CAM.pytorch)

## TODO

- [x] 预训练模型下载URL整理（参考Reference）；
- [ ] 模型的openvino格式的转换和对应的部署demo；
