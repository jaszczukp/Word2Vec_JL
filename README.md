# Word2Vec_JL

An implementation of the Word2Vec algorithm in Julia, supporting CBOW and Skip-Gram architectures with Negative Sampling and Hierarchical Softmax. Includes tools for training and evaluation on custom text

## Overview

The implementation supports both main Word2Vec architectures:
- Continuous Bag of Words (CBOW)
- Skip-Gram (SG)

and two optimization strategies:
- Hierarchical Softmax (HS)
- Negative Sampling (NS)

The code is modular and designed to offer full control over data preprocessing, vocabulary construction, training pair generation, model training, and evaluation.

## Features

- Text preprocessing and tokenization
- Construction of vocabulary and frequency tables
- Generation of training data using context windows and negative sampling
- CBOW and Skip-Gram models implemented from scratch
- Hierarchical Softmax with Huffman tree construction
- Negative Sampling with adjustable number of negative examples
- Custom training loop with tracking of loss function
- Cosine similarity-based evaluation of embedding quality
- Support for hyperparameter configuration:
  - Context window size
  - Embedding dimensionality
  - Learning rate
  - Number of training epochs
- Saving and loading model weights and embeddings


