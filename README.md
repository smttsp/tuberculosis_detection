# Tuberculosis Detection

This is an end-to-end tuberculosis detection tool which
contains 

- front-end
- model training
- back-end
  - cloud run
  - dockerized model
  - deployment of the model

## Demo

https://github.com/smttsp/tuberculosis_detection/assets/4594945/70501f04-af9e-49d8-8bd0-cb2fdf8bf738


## Installation

### Prerequisite: `pyenv`

https://github.com/pyenv/pyenv-installer

On macOS you can use [brew](https://brew.sh), but you may need to grab the `--HEAD` version for the latest:

```bash
brew install pyenv --HEAD
```

or

```bash
curl https://pyenv.run | bash
```

And then you should check the local `.python-version` file or `.envrc` and install the correct version which will be the basis for the local virtual environment. If the `.python-version` exists you can run:

```bash
pyenv install
```

This will show a message like this if you already have the right version, and you can just respond with `N` (No) to cancel the re-install:

```bash
pyenv: ~/.pyenv/versions/3.8.6 already exists
continue with installation? (y/N) N
```

### Prerequisite: `direnv`

https://direnv.net/docs/installation.html

```bash
curl -sfL https://direnv.net/install.sh | bash
```

### Developer Setup

If you are a new developer to this package and need to develop, test, or build -- please run the following to create a developer-ready local Virtual Environment:

```bash
direnv allow
python --version
pip install --upgrade pip
pip install poetry
poetry install
```

## Remarks

In a more professional settings, this repo should be split into multiple pieces. A good split may be

- model development repo
- inference + dockerization + deployment repo
- frontend

Because model development dependencies will make the inference pipeline unnecessarily larger. 
Similarly, frontend dependencies (currently `flask` and `flask-cors`) make the docker image larger.

To prevent this issue, I made all the model development libraries as `dev` dependencies. So that
they are not included in the docker image. But this is just a trick, not the preferred way.


## Data and Credit

The data for this model training is obtained from Kaggle:

https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset

Kudos to the team and their amazing work for collecting such a useful dataset!
