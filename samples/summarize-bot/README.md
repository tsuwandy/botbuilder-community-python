#### Git Issues Summarization Bot
Prerequisites:
Microsoft Visual C++ 14.0
Java

## Setup

This sample **requires** the Anaconda environment (which provides Jupyter Lab and other machine learning tools) in order to run.

The following instructions assume using the [Anaconda](<https://www.anaconda.com/>) environment (v4.6.11+). 

Note: Be sure to install the **64-bit** version of Anaconda for the purposes of this tutorial.

### Create and activate virtual environment

In your local folder, open an **Anaconda prompt** and run the following commands:

```bash
cd 101.corebot-bert-bidaf
conda create -n botsample python=3.6 anaconda -y
conda activate botsample # source conda 

conda install pytorch-cpu torchvision-cpu -c pytorch
pip install -r requirements.txt

python -m spacy download en
python -m spacy download en_core_web_lg

>>> import nltk
>>> nltk.download('punkt')

To run this sample:

```
python main.py
```

and you can test with the Bot Framework Emulator by connecting to http://localhost:9000
