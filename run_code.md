# How to run the code

### 1. Start Virtual Environment

```
$ source ~/.bashrc
$ workon traffic_signs
```

### 2. Change directories

```
(traffic_signs)$ cd traffic_sign_classification/code
```

### 3. Specify arguments:
There are 5 arguments that need to be added when runing the code. The `class Args()` can be seen in `arguments.py`. 

* `-m` or `--model` to add the path to the output model
* `-d`or `--dataset` to add the path to the input dataset
* `-i`or `--images` to add the path to the testing directory containing images
* `-pr`or `--predictions` to add the path to the output predictions directory
* `-pl`or `--plot` to add the path to the training history plot

### 4. Run line in shell

Depending on which database you want to train with, you can choose between only two traffic signs (turn right and turn left) or you can choose all 43. 

The default is set to only two traffic signs, which can be identified by the `_rl` addition. If you want to train the model with all 43 traffic signs, replace `_rl` with `_all`.

Default command for only two traffic signs:
```
(traffic_signs)$ python main.py --model ../output/neural_net.model --dataset ../gtsrb_rl --images ../gtsrb_rl/Test --predictions ../predictions_rl --plot ../output/plot.png
```

Command for all 43 traffic signs:
```
(traffic_signs)$ python main.py --model ../output/neural_net.model --dataset ../gtsrb_all --images ../gtsrb_all/Test --predictions ../predictions_all --plot ../output/plot.png
```

In case you want to train with the entire dataset, make sure to also change the paths within the `train.py` file. There are four lines marked with `#CHANGE`, all you need to do is uncomment/comment the lines you need


## Results

Let the code run through all epochs of the training process. The output in the shell will indicate the current state of the programme.

A classification report will appear with the prediction accuracy.

The precdiction will be continued and 30 images will be used to predict the correct label.

In the directory `traffic_sign_classification/predictions` the images can be viewed with their predicted labels.