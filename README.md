# Abstract of Thesis

This research evaluates Graph Convolutional Recurrent Network (GCRN) models—specifically
GConvGRU and GConvLSTM—against a baseline ARIMA model for predicting bike
transport demand for Pedal Me in different regions of London. Using the PyTorch Geometric
Temporal Library, the GCRN models were tested with different epochs and dropout levels, and
their performance was evaluated against the ARIMA model using MSE, RMSE and MAE in a
holdout validation. The results show that while GCRN models achieve a balance between training
and testing for short training periods, they overfit for longer training periods, resulting in
poor generalization. Despite the usefulness of dropout layers, they could not completely prevent
overfitting. The GCRN models did not perform better than the ARIMA baseline model
due to the small size of the dataset and the lack of features such as additional contextual information,
which prevented them from fully utilizing their capabilities in spatio-temporal modeling.
Future work should therefore investigate other regularization techniques and transfer learning
to improve performance. Furthermore, additional metrics such as connectivity should be
used to gain a better understanding of memory processing in GCRN models. In addition, the
extension of ARIMA to ARIMAX could improve the prediction of volatile fluctuations by
including further contextual information despite a smaller data set. Given the challenges faced
by the GCRN models, ARIMA is currently the better choice despite similar performance and
could serve as a decision support system for resource optimization in the respective regions of
Pedal Me in London.

**This repository consists of 2 Files:**
 1. "Code_TrafficForecastingGCRN.py" contains the python code with all machine learning algorithms
 2. "Written_Thesis_TrafficForecasting_Leon_Reiß.pdf" contains the written thesis submitted at Maastricht University (no publication restrictions)

**Algorithms are based on the following Paper:** Seo, Y., Defferrard, M., Vandergheynst, P., & Bresson, X. (2016). Structured Sequence Modeling
with Graph Convolutional Recurrent Networks. arXiv (Cornell University). https://doi.org/10.48550/arxiv.1612.07659
