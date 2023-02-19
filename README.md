## YOLOAir2☁️💡🎈 : Makes improvements easy again

基于 YOLOAir 算法库 ![](https://img.shields.io/github/stars/iscyy/yoloair?style=social)  : 👉[🔗 https://github.com/iscyy/yoloair](https://github.com/iscyy/yoloair)

手把手改进YOLO系列: 全面改进篇更新🔗：
[点击查看详情🚀 - 更多更全更新颖·《原创》·《最新》· 手把手改进YOLO系列详细教程(包括改进原理以及改进源代码 · 改进有效涨点)📚](https://github.com/iscyy/yoloair/wiki/Improved-tutorial-presentation)

--------

<div align="center">
  <p>
    <a align="center" href="https://github.com/iscyy/yoloair" target="_blank">
      <img width="850" src="https://img-blog.csdnimg.cn/d2e05e14ca32421ea4513051b8ce526b.png#pic_center"></a>
    <br><br>
  </p>
  <p>YOLOAir2 算法库是一个基于 PyTorch 的 YOLO 系列算法组合工具箱。统一模型代码框架、统一应用、统一改进、易于模块组合、构建更强大的网络模型。</p>

  简体中文 | [English](./README_EN.md)

  ![](https://img.shields.io/badge/News-2022-red)  ![](https://img.shields.io/badge/Update-YOLOAir-orange) ![](https://visitor-badge.glitch.me/badge?page_id=iscyy.yoloair)  

  #### **支持**

![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv5-red) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv7-brightgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv6-blueviolet) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOX-yellow) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOE-007d65) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv4-green) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-TOOD-6a6da9) 
![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOv3-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-YOLOR-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Scaled_YOLOv4-ff96b4) ![](https://img.shields.io/badge/Support-Transformer-9cf) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLO-lightgrey) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOv2-yellowgreen) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-PPYOLOEPlus-d5c59f) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-MLP-ff69b4) ![https://github.com/iscyy/yoloair](https://img.shields.io/badge/Support-Attention-green)

[特性🚀](#主要特性) • [使用🍉](#使用) • [文档📒](https://github.com/iscyy/yoloair2) • [报告问题🌟](https://github.com/iscyy/yoloair2/issues/new) • [更新💪](#-to-do) • [讨论✌️](https://github.com/iscyy/yoloair2/wiki) • [效果预览🚀](#效果预览)


![https://github.com/iscyy/yoloair](https://img-blog.csdnimg.cn/f7045ecc4f90430cafc276540dddd702.gif#pic_center)

</div>

## Introduction

☁️💡🎈YOLOAir2 is the second version of the YOLOAir series, The framework is based on YOLOv7, including YOLOv7, YOLOv6, YOLOv5, YOLOX, YOLOR, YOLOv4, YOLOv3, Transformer, Attention and Improved-YOLOv7... Support to improve Backbone, Neck, Head, Loss, IoU, NMS and other modules, As a perfection and addition of YOLOAir

**模型多样化**: 基于不同网络模块构建不同检测网络模型。

**模块组件化**: 帮助用户自定义快速组合Backbone、Neck、Head，使得网络模型多样化，助力科研改进检测算法、模型改进，网络排列组合🏆。构建强大的网络模型。

**统一模型代码框架、统一应用方式、统一调参、统一改进、集成多任务、易于模块组合、构建更强大的网络模型**。

内置集成YOLOv5、YOLOv7、YOLOv6、YOLOX、YOLOR、Transformer、PP-YOLO、PP-YOLOv2、PP-YOLOE、PP-YOLOEPlus、Scaled_YOLOv4、YOLOv3、YOLOv4、YOLO-Face、TPH-YOLO、YOLOv5Lite、SPD-YOLO、SlimNeck-YOLO、PicoDet等模型网络结构... 
集成多种检测算法 和 相关多任务模型 使用统一模型代码框架，**集成在 YOLOAir 库中，统一应用方式**。便于科研者用于论文算法模型改进，模型对比，实现网络组合多样化。包含轻量化模型和精度更高的模型，根据场景合理选择，在精度和速度俩个方面取得平衡。同时该库支持解耦不同的结构和模块组件，让模块组件化，通过组合不同的模块组件，用户可以根据不同数据集或不同业务场景自行定制化构建不同检测模型。

支持集成多任务，包括目标检测、实例分割、图像分类、姿态估计、人脸检测、目标跟踪等任务

<img src='https://img-blog.csdnimg.cn/1589c7f744004401b9d88132de35abe8.jpeg#pic_center' alt='ingishvcn'>

**Star🌟、Fork** 不迷路，同步更新。![](https://img.shields.io/github/stars/iscyy/yoloair?style=social)

项目地址🌟: https://github.com/iscyy/yoloair

### 主要特性🚀

🚀支持更多的YOLO系列算法模型改进(持续更新...)

YOLOAir 算法库汇总了多种主流YOLO系列检测模型，一套代码集成多种模型: 

- 内置集成 YOLOv5 模型网络结构、YOLOv7 模型网络结构、 YOLOv6 模型网络结构、PP-YOLO 模型网络结构、PP-YOLOE 模型网络结构、PP-YOLOEPlus 模型网络结构、YOLOR 模型网络结构、YOLOX 模型网络结构、ScaledYOLOv4 模型网络结构、YOLOv4 模型网络结构、YOLOv3 模型网络结构、YOLO-FaceV2模型网络结构、TPH-YOLOv5模型网络结构、SPD-YOLO模型网络结构、SlimNeck-YOLO模型网络结构、YOLOv5-Lite模型网络结构、PicoDet模型网络结构等持续更新中...

Todo
-----------

### 内置网络模型配置支持✨

🚀包括基于 YOLOv5、YOLOv7、YOLOX、YOLOR、YOLOv3、YOLOv4、Scaled_YOLOv4、PPYOLO、PPYOLOE、PPYOLOEPlus、Transformer、YOLO-FaceV2、PicoDet、YOLOv5-Lite、TPH-YOLOv5、SPD-YOLO等**其他多种改进网络结构等算法模型**的模型配置文件
______________________________________________________________________

### 效果预览🚀

|目标检测|目标分割|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/0b04579f80d145d7bd2e854753e9f367.jpeg' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/adb10e3c47e440f9acf4a183df9acf05.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|图像分类|实例分割|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/b1ca7795b70c4b6086b5e6b43b687c1b.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d29f6d6fa0624c5cacf107bd5d1a5fa2.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|目标分割|目标跟踪|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/0ce7c7584f2149c980d7e292fc1fcd24.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d9ae8953fb394a74a6b1096a401fc315.jpeg#pic_center' width='300px' height='180px'  alt='ingishvcn'>

|姿态估计|人脸检测|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/01f41103dc6c416aaeeb4577b87bb363.gif#pic_center' width='300px' height='260px' alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/d18a095621b64da69d2a712fa5613976.gif#pic_center' width='300px' height='260px'   alt='ingishvcn'>

|热力图01|热力图02|
:-------------------------:|:-------------------------:
<img src='https://img-blog.csdnimg.cn/eef8f911702242a5bb3e10a2e3188ca6.jpeg#pic_center' width='300px' height='180px' alt='ingishvcn'>  |  <img src='https://img-blog.csdnimg.cn/a22986632c25462cbe6abddc75a01ca5.jpeg#pic_center' width='300px' height='180px'   alt='ingishvcn'>


![yolo](https://img-blog.csdnimg.cn/b962fcd1bfa841399226ca564f22a385.gif#pic_center)
### 预训练权重🚀

- YOLOv5
https://github.com/ultralytics/yolov5/releases/tag/v6.1

- YOLOv4
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOv3
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOR
https://github.com/iscyy/yoloair/releases/tag/v1.0

- Scaled_YOLO
https://github.com/iscyy/yoloair/releases/tag/v1.0

- YOLOv7
https://github.com/iscyy/yoloair/releases/tag/v1.0

______________________________________________________________________

## 使用🍉

**About the code.** Follow the design principle of [YOLOv7](https://github.com/WongKinYiu/yolov7).  
The original version was created based on YOLOv7 and YOLOAir

### 安装

在**Python>=3.7.0** 的环境中克隆版本仓并安装 requirements.txt，包括**PyTorch>=1.7**。

```bash
$ git clone https://github.com/iscyy/yoloair2.git  # 克隆
$ cd yoloair2
$ pip install -r requirements.txt  # 安装
```

### 训练

```bash
$ python train.py --cfg configs/yolov5/yolov5s.yaml
```

### 推理

`detect.py` 在各种数据源上运行推理, 并将检测结果保存到 `runs/detect` 目录。

```bash
$ python detect.py --source 0  # 网络摄像头
                          img.jpg  # 图像
                          vid.mp4  # 视频
                          path/  # 文件夹
                          path/*.jpg  # glob
```
______________________________________________________________________

### Performance


______________________________________________________________________


### YOLOv7训练教程✨
与YOLOv5框架基本一致，可以参考[YOLOAir库](https://github.com/iscyy/yoloair)

- [训练自定义数据](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)  🚀 推荐
- [获得最佳训练效果的技巧](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)  ☘️ 推荐
- [使用 Weights & Biases 记录实验](https://github.com/ultralytics/yolov5/issues/1289)  🌟 新
- [Roboflow：数据集、标签和主动学习](https://github.com/ultralytics/yolov5/issues/4975)  🌟 新
- [多GPU训练](https://github.com/ultralytics/yolov5/issues/475)
- [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)  ⭐ 新
- [TFLite, ONNX, CoreML, TensorRT 导出](https://github.com/ultralytics/yolov5/issues/251) 🚀
- [测试时数据增强 (TTA)](https://github.com/ultralytics/yolov5/issues/303)
- [模型集成](https://github.com/ultralytics/yolov5/issues/318)
- [模型剪枝/稀疏性](https://github.com/ultralytics/yolov5/issues/304)
- [超参数进化](https://github.com/ultralytics/yolov5/issues/607)
- [带有冻结层的迁移学习](https://github.com/ultralytics/yolov5/issues/1314) ⭐ 新
- [架构概要](https://github.com/ultralytics/yolov5/issues/6998) ⭐ 新

______________________________________________________________________


### 未来增强✨
后续会持续建设和完善 YOLOAir 生态  
完善集成更多 YOLO 系列模型，持续结合不同模块，构建更多不同网络模型  
横向拓展和引入关联技术等等   

______________________________________________________________________

## Citation✨

```python
@article{2022yoloair2,
  title={{YOLOAir2}: Makes improvements easy again},
  author={iscyy},
  repo={github https://github.com/iscyy/yoloair2},
  year={2022}
}
```

## Statement
<details><summary> <b>Expand</b> </summary>

* The content of this site is only for sharing notes. If some content is infringing, please sending email.

* If you have any question, please discuss with me by sending email.
</details>

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
[https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)  
[https://github.com/iscyy/yoloair](https://github.com/iscyy/yoloair)

</details>
