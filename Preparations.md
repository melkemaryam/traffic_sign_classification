# Preparations

### 1. Install Virtualenv:

`$ pip3 install virtualenv virtualenvwrapper`

### 2. Edit `~/.bashrc` profile

`$ vim ~/.bashrc`
	
```	# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/Library/Python/3.8/bin/virtualenv
source $HOME/Library/Python/3.8/bin/virtualenvwrapper.sh
```
### 3. Source venv

`$ source ~/.bashrc`

### New terminal commands:

* Create an environment with `mkvirtualenv`
* Activate an environment (or switch to a different one) with `workon`
* Deactivate an environment with `deactivate`
* Remove an environment with `rmvirtualenv`

more infos [in the docs](https://virtualenvwrapper.readthedocs.io/en/latest/)

### 4. Create new venv:

`$ mkvirtualenv traffic_signs -p python3`

### 5. Install packages:

```
$ workon traffic_signs
$ pip install opencv-contrib-python
$ pip install numpy
$ pip install scikit-learn
$ pip install scikit-image
$ pip install imutils
$ pip install matplotlib
$ pip install tensorflow # or tensorflow-gpu
```

