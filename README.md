# YOLOX_Tracker

采用旷世的[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)的目标检测框和[KCF-DSST](https://github.com/liliumao/KCF-DSST)追踪算法，其中YOLOX采用TensorRT部署，详见之前的[repo](https://github.com/ChenJiahao031008/SLAM_YOLOv5)，配置和之前一致。数据关联采用IoU交并比计算。检测框和目标框融合目前采用比例系数法进行，后续考虑加入卡尔曼滤波进行优化。

目前该代码仍然在完善中。

demo版本的效果如下：

<img src="pic/demo.gif" style="zoom: 50%;" />
