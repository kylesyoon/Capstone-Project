# capstone_final_hand_detection
MLND Final Capstone Project - VIVA Hand Detection Challenge Using A Keras RetinaNet

## Setup

In order to get this project going, the [VIVA Hand Detection Challenge](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/) [training dataset](http://cvrr.ucsd.edu/vivachallenge/data/LISA_HD_Static.zip) and [evaluation kit](http://cvrr.ucsd.edu/vivachallenge/data/EvalTools_HD.zip) must be downloaded and extracted.

1. Download the training dataset.
2. Extract the contents into the `dataset/` directory.
3. Download the evaluation kit.
4. Extract the contents into the `evaluation/` directory.

The result should look like:

```
root/
	dataset/
		detectiondata/
			...
	evaluation/
		annotations/
			...
		curves/
			...
		...
```

Additionally, you'll need to download the [keras-retinanet](https://github.com/fizyr/keras-retinanet) into the root of the project as well.
Then from the */keras-retinanet/ directory* run:
```
python setup.py install --user
```
[(Step 2 of the Installation guide from the keras-retinanet README)](https://github.com/fizyr/keras-retinanet#installation)

