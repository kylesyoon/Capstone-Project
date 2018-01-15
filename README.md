# Capstone Project

## Udacity MLND

### VIVA Hand Detection Challenge Using A Keras RetinaNet

#### Overview

In this project, [RetinaNet](https://research.fb.com/publications/focal-loss-for-dense-object-detection/), a one-stage object detection system developed by Lin et al is used for the [VIVA Hand Detection Challenge](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/).

The model training piggy backs off the [`keras-retinanet`](https://github.com/fizyr/keras-retinanet) implementation.

You'll also need Octave and [Piotr's Computer Vision Matlab Toolbox](https://pdollar.github.io/toolbox/) for evaluation.

For more detail about this project, please refer to the [`capstone_report.pdf`](https://github.com/yoonapps/MLND-Capstone/blob/master/capstone_report.pdf).

#### Sample Detection

VIVA Test Data

![](https://github.com/yoonapps/MLND-Capstone/blob/master/misc/sample_1.png)

#### Evaluation

Some scores using the [VIVA Hand Detection Challenge evaluation kit](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/) available on the challenge website.

| VIVA Evaluation | L1 (AP/AR)  | L2 (AP/AR)   |
|-----------------|-------------|--------------|
| Epoch 18        |**92.6**/90.3| 82.3/71.1    |
| Epoch 21        | 91.0/90.7   | 78.9/**73.9**|

The results for the snapshot at Epoch 21 ranks **5th** in the leaderboard. (Ranked by L2 AR.)

#### Non-VIVA Challenge Detection

Not as good. Improvements can be made. (More image augmentation, larger dataset, etc)

![](https://github.com/yoonapps/MLND-Capstone/blob/master/misc/ironman_sample_3.png)
