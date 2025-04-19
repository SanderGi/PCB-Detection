# PCB Detection
![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
[![Licence](https://img.shields.io/github/license/SanderGi/PCB-Detection?style=flat-square)](./LICENSE)
[![maintenance-status](https://img.shields.io/badge/maintenance-passively--maintained-yellowgreen.svg?style=flat-square)](https://gist.github.com/taiki-e/ad73eaea17e2e0372efb76ef6b38f17b)
[![OBB mAP50](https://img.shields.io/badge/OBB_mAP50-93.0%25-green?style=flat-square)](#results)
[![Segmentation mAP50](https://img.shields.io/badge/SEG_mAP50-99.5%25-green?style=flat-square)](#results)
[![Object Detection mAP50](https://img.shields.io/badge/OBJ_DET_mAP50-99.5%25-green?style=flat-square)](#results)
[![Classification F1](https://img.shields.io/badge/CLS_F1-99.8%25-green?style=flat-square)](#results)

There are [a lot of models](https://universe.roboflow.com/roboflow-100/printed-circuit-board/model/3) for detecting components within a Printed Circuit Board (PCB), but not as many for detecting which pixels (if any) in an image contain the PCB itself. Being able to determine if and where a PCB is in an image is useful for [calculating its size to estimate carbon footprint]((https://github.com/SanderGi/LCA)), as a preprocessing step for detecting components, to limit the amount of image more expensive PCB defect detection models have to process, and more.

In this repo, we introduce a new dataset with **1000s of quality annotations** using data augmentation. We also present a <ins>**20+ percentage point improvement**</ins> over existing methods for Oriented Bounding Box (OBB) detection (2.6 MB, <ins>**93% mAP50**</ins>), Segmentation (6.4 MB, <ins>**99.5% mAP50**</ins>), Object Detection of axis aligned bounding boxes (6.4 MB, <ins>**99.5% mAP50**</ins>), and Classification (<ins>**99.8% F1 Score**</ins>) of PCBs using [YOLOv11](https://docs.ultralytics.com/models/yolo11/). We support an arbitrary number of PCBs in each image and are robust to occlusions, lighting conditions, camera settings, perspective, and non-PCB distractors.

## Usage
TIP: The models were trained with the PCBs making up <80% of the image. If you have images that are already closely cropped to the PCB, adding padding will yield better results.

### Oriented Bounding Box Detection (OBB)

1. Download [`the model weights`](./data/augmented_obb/runs/no_perspective3/weights/best.pt)
2. `pip install ultralytics`
3. Run the model with `yolo task=obb mode=predict model=[path to model weights] source=[path to test image]` from the terminal or with Python:

```python
from ultralytics import YOLO

model = YOLO('[path to model weights]')
results = model.predict('[path/to/test/image.jpg]')
```

### Segmentation

1. Download [`the model weights`](./data/augmented_seg/runs/no_perspective/weights/best.pt)
2. `pip install ultralytics`
3. Run the model with `yolo task=segment mode=predict model=[path to model weights] source=[path to test image]` from the terminal or with Python:

```python
from ultralytics import YOLO

model = YOLO('[path to model weights]')
results = model.predict('[path/to/test/image.jpg]')
```

### Object Detection and Classification
The Segmentation model also detects axis aligned bounding boxes. It can also be used to classify whether an image contains a PCB for the best results (99.8% F1 Score). For a smaller model (2.6 MB instead of 6.4 MB), the OBB model can also be used to classify whether an image contains a PCB (93.8% F1 Score). You could also compute axis aligned bounding boxes from the OBB model.

### Data
Download the data for your own projects:

- [Cropped PCB images](./data/cropped_pcbs): 2 GB
- [Background images](./data/backgrounds): 100 MB
- [Distraction images](./data/distractions): 27 MB
- [Augmented OBB images in YOLO dataset format](./data/augmented_obb): 13 GB
- [Augmented Segmentation images in YOLO dataset format](./data/augmented_seg): 13 GB

## Setup Development Environment

If you want to train/fine-tune your own models or generate more augmented training data, choose one of Pyenv or Conda to manage your Python environment.

0. `git clone https://github.com/SanderGi/PCB-Detection.git`
    - Make sure you have [git lfs](https://git-lfs.com/) installed and run `git lfs install` if you want to download dataset images

### With Pyenv

1. Install Python 3.10.12
    - [Install pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation)
    - Run `pyenv install 3.10.12`
    - Pyenv should automatically use this version in this directory. If not, run `pyenv local 3.10.12`
2. Create a virtual environment
    - Run `python -m venv ./venv` to create it
    - Run `. venv/bin/activate` when you want to activate it
        - Run `deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
3. Run the commands in './scripts/install.sh', e.g., with `. ./scripts/install.sh`. 
    - This will install dependencies and datasets. You should always activate your virtual environment `. ./venv/bin/activate` before running any scripts.

### With Conda

1. Install miniconda or anaconda
    - [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
    - Or [install anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create a virtual environment
    - Run `conda create --prefix ./venv python=3.10.12` to create it
    - Run `conda activate ./venv` when you want to activate it
        - Run `conda deactivate` when you want to deactivate it
    - Pro-tip: select the virtual environment in your IDE, e.g. in VSCode, click the Python version in the bottom left corner and select the virtual environment
3. Run the commands in './scripts/install.sh', e.g., with `. ./scripts/install.sh`. 
    - This will install dependencies and datasets. You should always activate your virtual environment `conda activate ./venv` before running any scripts. 

## Methods

### Data Preprocessing
We apply preprocessing to deduplicate PCBs and annotate masks/bounding boxes for the datasets such as Roboflow 100, FICS-PCB, and FCC without these annotations. The steps are documented in the `*_eda.ipynb` notebooks.

Noteworthy is the FCC "dataset". The Federal Communications Commission (FCC) keeps records of all devices with a radio transmitter sold in the US. This means we have tons of scanned documents with internal photos of PCBs from most manufacturers. These are very unstructured and not in an easily machine-readable format. We use the pipeline from https://github.com/SanderGi/LCA to prefilter the images for PCBs (90.32% accuracy) and draw preliminary bounding boxes (67.9% accuracy). We then manually correct the bounding boxes and remove duplicates.

The other datasets (Roboflow 100 and FICS-PCB) are easier to parse. All photos contain a PCB, are fairly well cropped, and don't contain any non-PCB objects. We use the [InSPyReNet](https://github.com/plemeri/transparent-background) model to remove the background and [a modified version](./scripts/correct4d.py) of an algorithm used to make 4D corrections when scanning a document into a PDF to detect the corners of the PCB, unwarp any weird perspective, and crop the PCB. One caveat is that Roboflow 100 contains a lot of duplicates both in terms of identical images and images of the same PCB rotated. We detect the rotated images and discard them. Then we remove exact duplicates. Finally everything is manually checked and corrected.

![unwarp](./data/unwarp.png)

### Pre-augmented Data Size
#### OBB and Segmentation
```text
Roboflow 100: 672 images -> deduplicate -> 194 pcbs
FICS-PCB: 30 pcbs
Micro PCB: (8.125k images of 13 pcbs) not annotated with OBB so not used yet
PCB-P: 16.6k images -> unlabeled so not used yet
FPIC: waiting for access from Zhihan
FCC internal photos: (100s of 1000s) -> select a few -> 46 pcbs
-- 
TOTAL: 270 pcbs
```

#### Object Detection [future work]
If we did not want the orientation of the PCB bounding box, we could include Micro PCB. It contains lots of images but has low pcb diversity so should be downsampled to ~200 images to avoid skewing the training distribution. This would give us a total of 470 images to augment. Parsing code can be found in [`notebooks/micropcb_eda.ipynb`](./notebooks/micropcb_eda.ipynb). Contributions with trained models are welcome!

#### Classification [future work]
Both the OBB and Segmentation models can be used for classification, but if we wanted even better performance we could train a dedicated classification model and include both Micro PCB, PCB-P, and a much larger selection of FCC images (since the annotation burden is much lower). Contributions with processed and manually verified datasets and/or trained models are welcome!

#### Using the Object Detection Model to Improve the OBB and Segmentation Models [future work]
Since more data is available for training Object Detection and Classification models, [they can be combined](https://docs.ultralytics.com/datasets/segment/#auto-annotation) with a general purpose segmentation model such as SAMv2 to automate labeling a larger dataset for OBB and Segmentation. Contributions with data labeling code and verified datasets are welcome!

### Data Augmentation

We run multiple types of data augmentation with different purposes. The full details are documented in [`./notebooks/augment.ipynb`](./notebooks/augment.ipynb).

#### Ahead of Time (5x)
We statically augment the dataset by taking the [`cropped pcbs`](./data/cropped_pcbs) and inserting them into a random **background** with random **scaling, rotation, and translation**. We also insert random non-pcb electronics and other objects into the background to teach the model to ignore distractions. We use this to create a dataset 5x the original size:

![augmented](./data/augment.png)

Since we have removed the background from the cropped pcbs in pre-processing, we can use the opacity to automatically create segmentation masks and apply the same transforms to the mask to get the augmented masks:

<img src="./data/augment_seg_img.png" alt="augmented image" width="120" style="float: left; margin-right: 10px;">
<img src="./data/augment_seg_mask.png" alt="augmented segmentation mask" width="120" style="float: left; margin-right: 10px;">
<div style="clear: both;"></div>

### On the Fly
To prevent the model from overfitting and to introduce further robustness to lighting/camera conditions, the presence of multiple PCBs, and occlusions/clipping, we use the following augmentations on the fly during training applied randomly each epoch:

1. **Multiple PCB Detection**: We use mosaic augmentation to combine multiple images into one. This is useful for detecting multiple PCBs in the same image. For Segmentation, we also randomly copy different masked out PCBs between images.
2. **Lighting/Camera/Weather Conditions**: We use random brightness, hue, saturation, color channel ordering, and blur filters to simulate different lighting/camera conditions
3. **Scale and Translation Pertubations**: We add random scale and translation perturbations to prevent the model from overfitting to the exact size and position of the PCBs. This also introduces cases where the PCB is partially occluded or clipped

OBB Sample Training Data:
![Augmented OBB Training Data](./data/augmented_obb/runs/no_perspective3/train_batch0.jpg)

Segmentation Sample Training Data:
![Augmented Segmentation Training Data](./data/augmented_seg/runs/no_perspective/train_batch1.jpg)

### Training
We set aside [a dedicated test set with 35 images](./data/augmented_obb/images/test) and split the rest of the data into 80% training and 20% validation. Training parameters can be found in [`notebooks/train_obb.ipynb`](./notebooks/train_obb.ipynb) and [`notebooks/train_seg.ipynb`](./notebooks/train_seg.ipynb). We fine-tune from pretrained checkpoints of YOLOv11. The OBB model has been pretrained on [DOTAv1](https://docs.ultralytics.com/datasets/obb/dota-v2/#dota-v10). The Segmentation model has been pretrained on [COCO](https://docs.ultralytics.com/datasets/segment/coco/). We don't fine-tune from a checkpoint that has been trained on PCB component segmentation to avoid test set leakage and overfitting.

![training curves](./data/augmented_obb/runs/no_perspective3/results.png)

### Results
Our models outperform all combinations of heuristics, Vision Language Models, general purpose segmentation, traditional CV approaches, and PCB component detection models as evaluated [here](https://github.com/SanderGi/LCA) (less than 70% F1 Score).

### OBB
Dataset    | Precision | Recall | F1 Score | mAP50  | mAP50-95
-----------|-----------|--------|----------|--------|---------
Training   | 100.0%    | 100.0% | 100.0%   | 100.0% | 100.0%
Validation | 100.0%    | 100.0% | 100.0%   | 99.5%  | 97.0%
Test       | 100.0%    | 88.4%  | 93.8%    | 93.0%  | 91.2%

Sample predictions:
![sample predictions](./data/augmented_obb/runs/no_perspective3/val_batch1_pred.jpg)

### Segmentation
Dataset    | Precision | Recall | F1 Score | mAP50 | mAP50-95
-----------|-----------|--------|----------|-------|---------
Training   | 100.0%    | 23.2%  | 37.7%    | 39.4% | 39.1%
Validation | 99.9%     | 39.6%  | 56.7%    | 51.7% | 51.0%
Test       | 99.7%     | 100%   | 99.8%    | 99.5% | 95.6%

Sample predictions:
![sample predictions](./data/augmented_seg/runs/no_perspective/val_batch1_pred.jpg)

### Object Detection*
_*These scores are for using the Segmentation model to detect axis aligned bounding boxes._

Dataset    | Precision | Recall | F1 Score | mAP50 | mAP50-95
-----------|-----------|--------|----------|-------|---------
Training   | 100.0%    | 23.2%  | 37.7%    | 39.4% | 39.3%
Validation | 99.9%     | 39.6%  | 56.7%    | 51.7% | 51.3%
Test       | 99.7%     | 100%   | 99.8%    | 99.5% | 94.5%

## Contributing
We welcome contributions! Take a look at the subsections of the [Methods Section](#methods) marked with `[future work]` for recommended areas of contribution. If you have any questions or suggestions, please open an issue or a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details. Feel free to use the code and models for your own projects. If you make something cool post it in an issue or email the [contact](#contact). Would love to see what you make!

## Contact
The contact for this project is [Alexander Metzger (alex@sandergi.com)](https://sandergi.com/vCard.vcf).

## References

### Data
- [FICS-PCB: A Multi-Modal Image Dataset for Automated Printed Circuit Board Visual Inspection](https://eprint.iacr.org/2020/366.pdf)
- [Roboflow 100: Printed Circuit Board Dataset](https://universe.roboflow.com/roboflow-100/printed-circuit-board)
- [FPIC: A Novel Semantic Dataset for Optical PCB Assurance](https://arxiv.org/pdf/2202.08414)
- [Micro PCB Images](https://www.kaggle.com/datasets/frettapper/micropcb-images/data)
- [PCB-P: Printed Circuit Boards at Perspectives](https://www.kaggle.com/datasets/benmalin/pcb-p-printed-circuit-boards-at-perspectives)
- [FCC Report Parsing](https://github.com/SanderGi/LCA)

### Data Augmentation
- [YOLO Data Augmentation Explained](https://rumn.medium.com/yolo-data-augmentation-explained-turbocharge-your-object-detection-model-94c33278303a)
- [YOLO Data Augmentation Docs](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [Oriented Bounding Box Dataset Format](https://docs.ultralytics.com/datasets/obb/)
- [Object Detection Dataset Format](https://docs.ultralytics.com/datasets/detect/)
- [Segmentation Dataset Format](https://docs.ultralytics.com/datasets/segment/)

### Training
- [YOLOv8 Training Docs](https://docs.ultralytics.com/modes/train/)
- [YOLOv11 pre-trained checkpoints](https://docs.ultralytics.com/models/yolo11/)
- [YOLO Object Detection Metrics](https://docs.ultralytics.com/guides/yolo-performance-metrics/)

### Inference
- [Ultralytics CLI](https://docs.ultralytics.com/usage/cli/)
