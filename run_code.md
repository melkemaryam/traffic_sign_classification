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

```
(traffic_signs)$ python main.py --model ../output/neural_net.model --dataset ../gtsrb-german-traffic-sign --images ../gtsrb-german-traffic-sign/Test --predictions ../predictions --plot ../output/plot.png
```
## Results

Let the code run through all epochs of the training process. The output in the shell will indicate the current state of the programme.

A classification report will appear with the prediction accuracy.

The precdiction will be continued and 30 images will be used to predict the correct label.

In the directory `traffic_sign_classification/predictions` the images can be viewed with their predicted labels.