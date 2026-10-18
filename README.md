# Traffic Forecasting with Graph Convolutional Recurrent Networks

**Forecasting bike delivery demand across London regions for Pedal Me: GCRN models (GConvGRU, GConvLSTM) compared with an ARIMA baseline.**

## Abstract

This thesis evaluates two Graph Convolutional Recurrent Network (GCRN) models, GConvGRU and GConvLSTM, against an ARIMA baseline for predicting bike transport demand for Pedal Me in different regions of London. The GCRN models were built with the PyTorch Geometric Temporal library and tested with varying numbers of epochs and dropout rates. All models were compared in a holdout validation using MSE, RMSE and MAE.

For short training periods, the GCRN models keep a good balance between training and test error. With longer training they overfit and generalise poorly, and dropout reduces this effect without fully preventing it. Overall, the GCRN models did not outperform the ARIMA baseline. The main reasons are the small dataset and the lack of additional contextual features, which kept the models from using their strength in spatio-temporal modelling.

Future work could explore further regularisation techniques and transfer learning to improve the GCRN models, and use additional metrics such as connectivity to better understand how these models process information over time. On the baseline side, extending ARIMA to ARIMAX could capture volatile fluctuations by adding contextual information, even with limited data. Given the current limitations of the GCRN models, ARIMA remains the better choice despite similar performance, and could serve as a decision support tool for resource planning across Pedal Me's regions in London.

## Key findings

- **Short training:** GCRN models balance training and test error well.
- **Long training:** GCRN models overfit; dropout helps but does not fully prevent it.
- **Comparison:** ARIMA performs on par with or better than the GCRN models on this dataset.
- **Bottleneck:** small dataset and missing contextual features, not the model architecture itself.

## Repository contents

| File | Description |
|---|---|
| `Code_TrafficForecastingGCRN.py` | Python code for all models (GConvGRU, GConvLSTM, ARIMA) and the evaluation |
| `Written_Thesis_TrafficForecasting_Leon_Reiß.pdf` | Full thesis submitted at Maastricht University (no publication restrictions) |

## References

- **Method:** Seo, Y., Defferrard, M., Vandergheynst, P., & Bresson, X. (2016). *Structured Sequence Modeling with Graph Convolutional Recurrent Networks.* arXiv. [doi.org/10.48550/arXiv.1612.07659](https://doi.org/10.48550/arXiv.1612.07659)
- **Library:** [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)
