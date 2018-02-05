## Implementation of [Asymmetric Tri-training for Unsupervised Domain Adaptation](https://arxiv.org/abs/1702.08400)


### Prerequisites
These codes work on Tensorflow 1.3.0.
### Download

Datasets are downloaded from svhn, [resized mnist train](https://drive.google.com/file/d/1UzKTtiRCkOq7vIrXVZkE8eWQTwf1Q4lN/view?usp=sharing)[resizesd mnist test](https://drive.google.com/file/d/15r597WzIbBcGR8ImyoBrvATjP8ulIRS1/view?usp=sharing), [synthetic digits](https://drive.google.com/file/d/0B-N5tVpsXW5mT2lvQmV6UE5uNFE/view?usp=sharing)(the dataset does not seem to be publicized by original author now).

### Run experiment
Fill the blank of dataset path in the following file.
```
python svhn_mnist_train.py
```