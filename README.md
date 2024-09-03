# pcretrieval
Model retrieval and pose estimation.

## Data
### ShapeNet
[Download](https://drive.google.com/drive/folders/1G0rf-6HSHoTll6aH7voh-dXj6hCRhSAQ) From [PointFlow](https://github.com/stevenygd/PointFlow)

## ScanNet, Scan2Cad
Mount the storage "scannet".

## Usage

### Train
Pretrained mode from FCGF is needed. [link](https://github.com/chrischoy/FCGF). Please put it under ```pcretrieval/ckpts```.<br>
Logs will be saved to ```pcretrieval/logs/LOG_NAME.txt``` and checkpoints will be saved to ```pcretrieval/ckpts/CKPT_NAME```. <br>
Default configuration options are in ```config.py```.

Use ```train.py``` for training.

### Evaluation
Use ```evaluation.py``` to evaluate retrieval or pose registration module.

Use ```Scan2CAD_eval.py``` and ```Shapenet_eval.py``` for the evaluation of the whole pipeline. ```scan2cad_eval.ipynb``` and ```shapenet_eval.ipynb``` are the jupyter version of evaluation code.

### Other
```category_dist.py``` and ```scan2cad_dist.py``` compute the pair-wise chamfer distance given a dataset of point cloud.

```eval_example.ipynb``` and ```example.ipynb``` give example of running the evaluation code and training code. Not useful now, given that we have a set of cluster commands.

```scene_level.ipynb``` produce the scene-level alignment and visualization result.

```crop_scan.py```

```visualize_scannet.py``` aligns the labeled CAD model with the scan and visualize.

```sym_labeling.ipynb``` automatically label the rotation symmetry.


### Command example

See the readme in the cluster package.

## Code structure
    .
    ├── config                   # Data split and labels
    ├── datasets                 # Dataset definition for ShapeNet and Scan2CAD
    ├── models                   # Network definition
    ├── test_time                # Test time class that can run the whole pipeline
    ├── trainer                  # Trainers
    ├── utils                    # Utility functions including data I/O, preprocessing, retrieval, registration, symmetry and etc.
    └── other sripts
