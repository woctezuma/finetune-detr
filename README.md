# Finetune DETR

The goal of this [Google Colab](https://colab.research.google.com/) notebook is to fine-tune Facebook's DETR (DEtection TRansformer).

## Usage

-   Acquire a dataset, e.g. the the `balloon` dataset,
-   Convert the dataset to the COCO format,
-   Run `finetune_detr.ipynb` to fine-tune DETR on this dataset.
-   (WIP) Alternatively, run `finetune_detectron2.ipynb` to rely on the detectron2 wrapper.

## Data

DETR will be fine-tuned on a tiny dataset: the [`balloon` dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon).
We refer to it as the `custom` dataset.

We expect the directory structure to be the following:
```
path/to/coco/
├ annotations/  # JSON annotations
│  ├ annotations/custom_train.json
│  └ annotations/custom_val.json
├ train2017/    # training images
└ val2017/      # validation images
```

NB: if you are confused about the number of classes, check [this Github issue](https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223).

## References

-   Facebook's [DETR](https://github.com/facebookresearch/detr)
-   My [fork](https://github.com/woctezuma/detr/tree/finetune) of DETR to fine-tune on a dataset with a single class
-   My [fork](https://github.com/woctezuma/VIA2COCO/tree/fixes) of VIA2COCO to convert annotations from VIA format to COCO format
-   An [official notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb) showcasing DETR
-   An [official notebook](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb) showcasing the COCO API
-   An [official notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) showcasing the Detectron2 wrapper for DETR
-   A [Github Gist](https://gist.github.com/mlk1337/651297e28199b4bb7907fc413c49f58f) explaining how to fine-tune DETR
-   A [Github issue](https://github.com/facebookresearch/detr/issues/9#issuecomment-636391562) explaining how to load a fine-tuned DETR
-   A [Github issue](https://github.com/facebookresearch/detr/issues/9) discussing the fine-tuning of DETR
-   [DETR checkpoints](https://github.com/facebookresearch/detr#model-zoo): remove the classification head, then fine-tune
-   [Detectron2 wrapper](https://github.com/facebookresearch/detr/tree/master/d2) for DETR ; caveat: this wrapper only supports box detection.
-   A [blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about another approach (Mask R-CNN) and the `balloon` dataset    

<!-- Definitions -->
