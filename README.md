# master-thesis-jakub-simon



## Running a new training
The LSTM training can be started by running:

```
python .\main_LSTM.py
```

all the hyperparameters are set in the Config dictionary in the python file.

the Transformer can be training by running:

```
python .\main.py
```

## Pretrained weights
The file summarizer.h5 contains the best weights for the LSTM, the file DistilBART.h5 contains the best weights for the Transformers-based approach

## Inference
```
usage: inference.py [-h] [-l] [--use_text_rank] [-w WEIGHTS] [--lstm_weights LSTM_WEIGHTS] [-t TEXT] [-k TOP_K]

Seq2Seq Inference

optional arguments:
  -h, --help            show this help message and exit
  -l, --use_lstm        Uses lstm as preselector
  --use_text_rank       Uses lstm as preselector
  -w WEIGHTS, --weights WEIGHTS
                        Path to summarizer weights
  --lstm_weights LSTM_WEIGHTS
                        Path to lstm weights
  -t TEXT, --text TEXT  Path to the text file to be summarized
  -k TOP_K, --top_k TOP_K
                        Number of sentences to be summarized
```

when using textrank or lstm top-k needs to be set, when using LSTM the lstm weights have to be set and when using textrank and lstm together text rank is ran first.