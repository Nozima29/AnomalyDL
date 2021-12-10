## Anomaly Detection with DL models

### Tested models:
*   seq2seq (-> decoder with attention)
*   AutoEncoder(->LSTM encoder & decoder layers)
*   <b>Transformer</b> (-> Encoder & Decoder with Attention)

### Dataset:
*   SWAT dataset (not uploaded in this repo due to large size)

### Files description

    parser.py

Initial data preparation/preprocessing and normalization, converts .csv data to .dat format

    swat_dataset.py

Class extends Torch Dataset to create train and test dataset with sliding window size

    encoder.py

Implementation of LSTMAutoEncoder model

    seq2seq.py

Implementation of Seq2Seq model

    transformer.py

Implementaion of TransformerEncoder model

    train.py

training all models, saved models are saved in checkpoints/ folder

    evaluate.py

evaluating model with ground truth values (labels)

    utils.py

helper functions for plotting graphs and calculating ROC-AUC results

### All models are implemented with <b>Pytorch</b> DL library