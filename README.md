# Sentiment Analysis on IMDB Reviews using LSTM and Keras 

Sentiment analysis is a text classification technique used to determine emotions, such as positive and negative sentiments in text data. In this project, we leverage the power of LSTM (Long Short-Term Memory), a recurrent neural network (RNN) architecture, to perform sentiment analysis. We implement this using the Keras deep learning library, which can be executed on the TensorFlow backend.

## Dataset

For this project, we utilize the [50,000 IMDB Movie Reviews dataset from Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). This dataset comprises two columns: movie reviews and sentiments (positive and negative). Notably, the dataset is evenly split between positive and negative sentiments.

## Architecture

Our model consists of the following key layers:

1. **Embedding Layer**
2. **LSTM Layer**
3. **Dense Layer (with Sigmoid Activation)**

We employ the Adam optimizer and use Binary Crossentropy as the loss function for training.

## References

- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [GitHub: Sentiment Analysis using LSTM with Keras](https://github.com/sanjay-raghu/sentiment-analysis-using-LSTM-keras/blob/master/lstm-sentiment-analysis-data-imbalance-keras.ipynb)
- [Towards Data Science: Sentiment Analysis using LSTM](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948)
- [Kaggle: LSTM Sentiment Analysis with Keras](https://www.kaggle.com/ngyptr/lstm-sentiment-analysis-keras)
- [YouTube: Sentiment Analysis with LSTM](https://www.youtube.com/watch?v=qpb_39IjZA0)
- [Illustrated Guide to LSTMs and GRUs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)
- [Understanding Keras Dense Layers](https://medium.com/@hunterheidenreich/understanding-keras-dense-layers-2abadff9b990)

## Usage

You can use this code to perform sentiment analysis on text data. Make sure to follow the steps outlined in the project.

