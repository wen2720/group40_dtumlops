# group40_leaf

MLOps 

## Project structure

The directory structure of the project looks like this:

```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Project description

#### Goal of the Project
The primary objective of this project is to set up and organize a machine learning operations (MLOps) framework to apply state-of-the-art machine learning techniques that we learned throughout the course. The project aims to recognize and categorize leaf species.

#### Framework
To achieve the goal of the project, we have selected the [TIMM (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models) framework. TIMM has a big collection of pre-trained models. It allows us to use the provided models in order to be more focused on organizing and setting up the project.  

#### Data
The data we selected is the [Leaf Classification dataset](https://www.kaggle.com/c/leaf-classification/data?select=test.csv.zip) from Kaggle and based on it a classification problem is aimed. The dataset consists of approximately 1,584 images of leaf specimens (16 samples each of 99 species),  the images are black and white with leaves being black on a white background. The classification will be based on the 3 digital features of the leaf images: margin, shape, and texture. The goal is network to detect the value of the three features and classify the leaf through the feature value.

#### Model
CNN architecture with powerful learning ability is considered the model we mainly explore. After some research, we decided to use the ConvNeXt model which has huge potential in image learning.


### 1. Local run of the training.py file.
$python src/group40_leaf/train.pY

### 2. Build docker image locally by train.dockerfile 
$docker build -f dockerfiles/train.dockerfile . -t leaf-mlops:latest

### 3. Run docker container locally by the following command 

3.1 Linux:
$docker run --rm -it -v $(pwd)/models:/models/ -v $(pwd)/data:/data/ leaf-mlops:latest data/

3.2 MacOS:
$docker run --rm -it -v ${PWD}/models:/models/ -v $(pwd)/data:/data/ leaf-mlops:latest

3.3 Windows:
$docker run --rm -it -v %cd%/models:/models/ -v %cd%/data:/data/ leaf-mlops:latest