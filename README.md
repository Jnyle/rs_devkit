# Devkit For Remote Sensing
![demo image](DOTA_devkit/demo/demo.png)
## Introduction
function
- [x] HorizontalDOTA2COCO(DOTA2COCO.py)
- [x] RotatedDOTA2COCO(DOTA2COCOROTATED.py)
- [x] Data select\view(DOTA.py)
- [x] Data crop(ImgSplit_multi_process.py \ SplitOnlyImage_multi_process.py)
- [x] ResultMerge(ResultMerge_multi_process.py)
- [x] Evaluate(dota_evaluation_task1.py) 
- [x] coco2coco
- [x] coco2txt
- [x] coco2xml
- [x] csv2coco
- [x] csv2xml
- [x] delete_gt_too_much
- [x] kmeans
- [x] mAP_evaluate
- [x] trick
- [x] txt2coco
- [x] txt2xml
- [x] SmallObjectAug


```shell
export INSTALL_DIR=$PWD
git clone git@gitlab.bj.sensetime.com:yanhongchang/rs_devkit.git
cd rs_devkit/DOTA_devkit
python setup.py install
unset INSTALL_DIR
```