# SciCite-Team4

This repository contains the dataset and code to address the classification task working with the SciCite dataset. 

Aim: Given an input citation sentence (“context”), classify its intent as one among {background, method, comparison}.

For details on the approach, refer to our report that is included in the repository.

The following directories contains the code to the respective experiments: 
1. attention: This directory contains the code comparing uniLSTM with and without self-attention mechanism. 
2. BERT_models: This directory contains (1) code for the baseline bert model and with sectionName as a feature (2) code for the SciBERT model (3) code for the SciBERT embeddings + LSTM model
3. traditional_model: This directory contains the code to all the traditional models we experimented with. 
4. Word2Vec_LSTM: This directory contains the code to feature selection with LSTM, and comparison with biLSTM. 