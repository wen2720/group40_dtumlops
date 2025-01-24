# group40_leaf

MLOps 

## Project structure

The directory structure of the project looks like this:

```txt
group40_dtumlops
├── .dvc/
│   ├── .gitignore            # DVC
│   └── config
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── .idea/                    # Idea
│   ├── .gitignore
│   ├── misc.xml
│   ├── mlops.iml
│   ├── modules.xml
│   ├── vcs.xml
│   └── inspectionProfiles/
│       └── profiles_settings.xml
├── configs/                  # Configuration files
│   ├── .gitkeep
│   └── sweep.yaml
├── data/                     # Data directory
│   ├── images/
│   |   ├── 1.jpg
│   |   ├── ...
│   |   └── 1584.jpg
│   ├── processed
│   ├── raw
│   ├── test.csv
│   └── train.csv
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   ├── README.md
│   └── source/
│       └── index.md
├── models/                   # Trained models
│   └── .gitkeep
├── notebooks/                # Jupyter notebooks
│   └── .gitkeep
├── reports/                  # Reports
│   ├── .gitkeep
│   ├── README.md
│   ├── report.py
│   └── figures/
│       ├── .gitkeep
│       ├── artifacts_100.png
│       ├── bucket.png
│       ├── build.png
│       ├── overview.png
│       ├── profiler_tensorboard.png
│       ├── registry.png
│       ├── sweep_100.png
│       ├── w&b_100.png
│       └── wandb.png
├── src/                      # Source code
│   ├── group40_leaf/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── test.py
│   │   ├── train.py
│   │   └── visualize.py
│   └── __init__.py
├── tests/                    # Tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   └── integrationstests/
│       └── test_api.py
├── wandb/                    # weight & biases
│   ├── latest-run/
│   ├── run-... /
│   └── sweep-... /
├── .coveragerc
├── .dvcignore
├── .gitignore
├── .pre-commit-config.yaml
├── cloudbild.yaml
├── data.dvc
├── Dockerfile
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements_dev.txt      # Development requirements
├── requirements_tets.txt     # Test requirements
├── requirements.txt          # Project requirements
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
```
$python src/group40_leaf/train.py
```

### 2. Build docker image locally by train.dockerfile 
```
$docker build -f dockerfiles/train.dockerfile . -t leaf-mlops:latest
```

### 3. Run docker container locally by the following command 

3.1 Linux:
`$docker run --rm -it -v $(pwd)/models:/models/ -v $(pwd)/data:/data/ leaf-mlops:latest`

3.2 MacOS:
`$docker run --rm -it -v ${PWD}/models:/models/ -v $(pwd)/data:/data/ leaf-mlops:latest`

3.3 Windows:
`$docker run --rm -it -v %cd%/models:/models/ -v %cd%/data:/data/ leaf-mlops:latest`