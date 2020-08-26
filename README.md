# Finetune DETR

The goal of this [Google Colab](https://colab.research.google.com/) notebook is to fine-tune Facebook's DETR (DEtection TRansformer).

<img alt="With pre-trained DETR" src="https://github.com/woctezuma/finetune-detr/wiki/img/pretrained_detr.jpg" width="375"> -> <img alt="With finetuned DETR" src="https://github.com/woctezuma/finetune-detr/wiki/img/finetuned_detr.jpg" width="375">

From left to right: results obtained with pre-trained DETR, and after fine-tuning on the `balloon` dataset.

## Usage

-   Acquire a dataset, e.g. the the `balloon` dataset,
-   Convert the dataset to the COCO format,
-   Run [`finetune_detr.ipynb`][finetune_detr-notebook] to fine-tune DETR on this dataset.
-   Alternatively, run [`finetune_detectron2.ipynb`][finetune_detectron2-notebook] to rely on the detectron2 wrapper.

NB: Fine-tuning is recommended if your dataset has [less than 10k images](https://github.com/facebookresearch/detr/issues/9#issuecomment-635357693).
Otherwise, training from scratch would be an option.

## Data

DETR will be fine-tuned on a tiny dataset: the [`balloon` dataset](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon).
We refer to it as the `custom` dataset.

There are 61 images in the training set, and 13 images in the validation set.

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

## Metrics

Typical metrics to monitor are shown in [this notebook][metrics-notebook].
These include:
-   losses (total loss, classification loss, l1 bbox distance loss, GIoU loss,
-   errors (cardinality error, class error),
-   the Average Precision (AP), which is [the primary challenge metric](https://cocodataset.org/#detection-eval) for the COCO dataset.

As mentioned in [the paper](https://arxiv.org/abs/2005.12872), there are 3 components to the matching cost and to the total loss:
-   classification loss,
```python
def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    [...]
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
    losses = {'loss_ce': loss_ce}
```
-   l1 bounding box distance loss,
```python
def loss_boxes(self, outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h),normalized by the image
       size.
    """
    [...]
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes
```
-   [Generalized Intersection over Union (GIoU)](https://giou.stanford.edu/) loss, which is scale-invariant.
```python
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
```

Moreover, there are two errors:
-   cardinality error,
```python
def loss_cardinality(self, outputs, targets, indices, num_boxes):
    """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty
    boxes. This is not really a loss, it is intended for logging purposes only. It doesn't
    propagate gradients
    """
    [...]
    # Count the number of predictions that are NOT "no-object" (which is the last class)
    card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    losses = {'cardinality_error': card_err}
```
-   class error,
```python
    # TODO this should probably be a separate loss, not hacked in this one here
    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
```
where [`accuracy`](https://github.com/facebookresearch/detr/blob/5e66b4cd15b2b182da347103dd16578d28b49d69/util/misc.py#L432) is:
```python
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
```

## Results

You should obtain acceptable results with 10 epochs, which require a few minutes of fine-tuning.

Out of curiosity, I have over-finetuned the model for 300 epochs (close to 3 hours).
Here are:
-   the last [checkpoint][checkpoint-300-epochs] (~ 500 MB),
-   the [log file][log-300-epochs].

All of the validation results are shown in [`view_balloon_validation.ipynb`][view-validation-notebook].

## References

-   Official repositories:
    - Facebook's [DETR](https://github.com/facebookresearch/detr) (and [the paper](https://arxiv.org/abs/2005.12872))
    - Facebook's [detectron2 wrapper for DETR](https://github.com/facebookresearch/detr/tree/master/d2) ; caveat: this wrapper only supports box detection
    - [DETR checkpoints](https://github.com/facebookresearch/detr#model-zoo): remove the classification head, then fine-tune
-   My forks:
    - My [fork](https://github.com/woctezuma/detr/tree/finetune) of DETR to fine-tune on a dataset with a single class
    - My [fork](https://github.com/woctezuma/VIA2COCO/tree/fixes) of VIA2COCO to convert annotations from VIA format to COCO format
-   Official notebooks:
    - An [official notebook](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb) showcasing DETR
    - An [official notebook](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb) showcasing the COCO API
    - An [official notebook](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5) showcasing the detectron2 wrapper for DETR
-   Tutorials:
    - A [Github issue](https://github.com/facebookresearch/detr/issues/9) discussing the fine-tuning of DETR
    - A [Github Gist](https://gist.github.com/woctezuma/e9f8f9fe1737987351582e9441c46b5d) explaining how to fine-tune DETR
    - A [Github issue](https://github.com/facebookresearch/detr/issues/9#issuecomment-636391562) explaining how to load a fine-tuned DETR
-   Datasets:
    - A [blog post](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about another approach (Mask R-CNN) and the [`balloon`](https://github.com/matterport/Mask_RCNN/tree/master/samples/balloon) dataset
    - A [notebook](https://github.com/matterport/Mask_RCNN/blob/master/samples/nucleus/inspect_nucleus_model.ipynb) about the [`nucleus`](https://github.com/matterport/Mask_RCNN/tree/master/samples/nucleus) dataset

<!-- Definitions -->

[pretrained-detr-image]: <https://github.com/woctezuma/finetune-detr/wiki/img/pretrained_detr.jpg>
[training-loss-image]: <https://github.com/woctezuma/finetune-detr/wiki/img/loss_finetuning_detectron2.jpg>
[finetuned-detr-image]: <https://github.com/woctezuma/finetune-detr/wiki/img/finetuned_detr.jpg>

[finetune_detr-notebook]: <https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb>
[finetune_detectron2-notebook]: <https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detectron2.ipynb>
[view-validation-notebook]: <https://colab.research.google.com/github/woctezuma/finetune-detr/blob/view-validation/view_balloon_validation.ipynb>

[checkpoint-300-epochs]: <https://drive.google.com/file/d/1BCtf4FxHl7F9ZJjxJ_lXymg_DAOxsMJQ/view?usp=sharing>
[log-300-epochs]: <https://drive.google.com/file/d/13wkKqRikEwjrDARaLg88qt7uJsk_cZzQ/view?usp=sharing>

[metrics-notebook]: <https://colab.research.google.com/github/lessw2020/Thunder-Detr/blob/master/View_your_training_results.ipynb>
