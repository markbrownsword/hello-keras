## Python Machine Learning (using Keras and Tensorflow)
Introduction to machine learning with Keras

### Prerequisites
[Install Miniconda](https://conda.io/miniconda.html)  
[Install PyCharm](https://www.jetbrains.com/pycharm/download/#section=linux)

```bash
# Install source from Github
mkdir HelloKeras  
cd HelloKeras  
git clone <https://github.com/markbrownsword/hello-keras.git> .  
```

### Setup

```bash
# Create Python 3.5 environment
conda create --name python35 --file requirements.txt

# Activate environment
conda activate python35

# Install Tensorflow
pip install tensorflow

# Deactivate environment
conda deactivate
```

### Run
Open HelloKeras in PyCharm and set project interpreter to python35 conda environment  
Run `binary_classifier_harness.py`