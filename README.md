# No Regrets

We build a sensitive information detection platform, designed as a Single Page Application, which effectively integrates front-end technologies with a Python-based back-end using Flask. This setup facilitates real-time text analysis, allowing users to immediately identify and manage sensitive or toxic content. By providing immediate, clear feedback, the platform empowers users to make informed decisions about their digital communications.

https://github.com/yunbinmo/cs5246-project-no-regrets/assets/77217780/8700776a-0a71-425d-bd21-435dca5521db

## Prerequisites


Before you begin, ensure you have met the following requirements:
- You have installed Python 3.10.

## Installation

To install the necessary packages, follow these steps:

```bash
python -m pip install -r requirements.txt
```

Downloading the Pre-trained Model:

We have trained a model that you can directly use for your convenience. Follow the steps below to download the pre-trained model:

Visit this Google Drive link: [Pre-trained Model](https://drive.google.com/drive/folders/1A67bpS8tLUfaB7-ePzvMWIHg88hFgFQY?usp=sharing)

Download all the model checkpoints found in the Google Drive folder and put in at the same directory as `app.py`

## Running

To start up the flask app locally, run:

```bash
python app.py
```
