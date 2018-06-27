# Capstone Project

## Udacity MLND

### VIVA Hand Detection Challenge Using A Keras RetinaNet

### Overview

In this project, [RetinaNet](https://research.fb.com/publications/focal-loss-for-dense-object-detection/), a one-stage object detection system developed by Lin et al is used for the [VIVA Hand Detection Challenge](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/).

The model training piggy backs off the [`keras-retinanet`](https://github.com/fizyr/keras-retinanet) implementation.

You'll also need Octave and [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/) for evaluation.

For more detail about this project, please refer to the [`capstone_report.pdf`](https://github.com/yoonapps/MLND-Capstone/blob/master/capstone_report.pdf).

### Setup

1. Clone this repository.
2. Download and extract the VIVA Hand Detection Challenge [dataset](http://cvrr.ucsd.edu/vivachallenge/data/LISA_HD_Static.zip).
3. Download and extract the VIVA Hand Detection Challenge [evaluation kit](http://cvrr.ucsd.edu/vivachallenge/data/EvalTools_HD.zip).
4. Create environent with the `environment.yml` file. *(Optional)*
5. In the `preprocessing.ipynb` update the directories to properly point to the downloaded dataset. (The test data annotations are in the evaluation kit.)
6. Download Octave or MATLAB.
7. Download [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/).
8. Update the path in the `demo.m` file in the evaluation kit downloaded at step 3 to point to the toolbox downloaded above.
9. The evaluation kit file `main_handdetect.m` did not work for me. If you're having the same issue, try changing :
```
dt{currloc} = [dt{currloc};currbb];
```
to 
```
[dt{currloc}] = deal([dt{currloc};currbb]);
```
on lines 46 and 70.

### Sample Detection

VIVA Test Data

![](https://github.com/yoonapps/MLND-Capstone/blob/master/misc/sample_1.png)

### Evaluation

Some scores using the [VIVA Hand Detection Challenge evaluation kit](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/) available on the challenge website.

| VIVA Evaluation | L1 (AP/AR)  | L2 (AP/AR)   |
|-----------------|-------------|--------------|
| Epoch 18        |**92.6**/90.3| 82.3/71.1    |
| Epoch 21        | 91.0/90.7   | 78.9/**73.9**|

The results for the snapshot at Epoch 21 ranks **4th** in the leaderboard. (Ranked by L2 AR.)

### Non-VIVA Challenge Detection

Not as good. Improvements can be made. (More image augmentation, larger dataset, etc)

![](https://github.com/yoonapps/MLND-Capstone/blob/master/misc/ironman_sample_3.png)
