# CNN-MLP-GAF-Deep-Learning-Finance

> **Deep Learning for Finance**

> Applying MLP and CNN-GAF architectures to Bitcoin price forecasting and direction classification.

---

## Overview

This project applies deep learning models to financial time series using Bitcoin (BTC-USD) daily closing prices from January 2018 to June 2023. The core question is: **does the way you preprocess financial data affect deep learning model performance?**

Three data representations are compared across two model architectures:

| Representation          | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| Price Levels            | Raw closing prices - non-stationary                         |
| Log-Returns             | ln(Pₜ / Pₜ₋₁) - stationary transformation                   |
| Fractional Differencing | Lopez de Prado method - stationary with memory preservation |

---

## Methodology

### Step 1 - Time Series Analysis

This section analyzes the Bitcoin (BTC-USD) price series using three different representations.

#### Step 1a - Raw Price Series (Levels)

The raw BTC-USD price series exhibits strong non-stationarity with a clear upward trend over the analysis period. The ADF test confirms non-stationarity (p-value > 0.05), indicating the presence of a unit root. The ACF shows very slow decay, characteristic of a random walk process. The distribution is highly right-skewed with heavy tails, reflecting Bitcoin's explosive growth periods.

#### Step 1b - Stationary Transformation (Log-Returns)

Log-returns are computed as r*t = ln(P_t / P*{t-1}). Log-returns successfully transform the series into a stationary representation, as confirmed by the ADF test (p-value < 0.05). The series fluctuates around zero with no discernible trend. However, this transformation removes all memory of the original price levels, potentially discarding valuable long-term information. The distribution shows excess kurtosis, indicating fat tails typical of financial returns.

#### Step 1c - Fractional Differencing

Fractional differencing is implemented using the fixed-width window method (Lopez de Prado, _Advances in Financial Machine Learning_, Ch. 5). The optimal differencing order d\* is found by searching for the minimum d where ADF p-value < 0.05. This approach achieves stationarity while maintaining a higher correlation with the original price levels compared to log-returns, suggesting better preservation of long-term dependencies.

#### Step 1d - Comparative Commentary

| Representation          | Stationarity   | Memory  | Suitable For             |
| ----------------------- | -------------- | ------- | ------------------------ |
| Price Levels            | Non-stationary | Full    | Derivative pricing only  |
| Log-Returns             | Stationary     | None    | Short-term forecasting   |
| Fractional Differencing | Stationary     | Partial | Long-horizon forecasting |

---

### Step 2 - Multi-Layer Perceptron (MLP)

Each MLP uses a rolling window of 10 lags to predict the next value (regression task). An 80/20 train-test split preserving time order is applied. MinMaxScaler normalizes inputs. Architecture: Dense(64) → Dropout → Dense(32) → Dropout → Dense(1). Early stopping prevents overfitting.

#### Step 2a - MLP on Price Levels

Training on non-stationary price levels produces misleadingly good R-squared values because the model learns to predict the trend rather than the actual dynamics. This approach fails to generalize as the model captures spurious correlations inherent in random walk processes.

#### Step 2b - MLP on Log-Returns

Using stationary log-returns provides more reliable forecasts. The model learns the actual return dynamics rather than trend-following behavior. The lower R-squared is preferable as it reflects realistic forecasting capability in financial markets.

#### Step 2c - MLP on Fractionally Differenced Series

This approach balances stationarity with memory preservation. Results typically fall between the levels and log-returns models, capturing some long-term dependencies while maintaining stationarity.

#### Step 2d - MLP Evaluation

Non-stationary inputs create unreliable forecasts that appear accurate but fail in practice. Stationary transformations are essential for robust model generalization. For short-term forecasting, log-returns provide the most stable and generalizable results. For longer horizons where memory effects matter, fractional differencing offers a better balance.

---

### Step 3 - CNN with Gramian Angular Field (GAF)

Time series windows are encoded into 2D image representations using Gramian Angular Field transformation (via `pyts`). The classification objective is to predict whether the next price will be higher (1) or lower/equal (0) than the current price. CNN architecture: Conv2D → MaxPooling2D → Conv2D → MaxPooling2D → Flatten → Dense(64) → Dropout → Dense(1, sigmoid).

#### Step 3a - CNN on Price Levels with GAF

The CNN trained on GAF images from price levels shows moderate performance. The non-stationary nature of the underlying data introduces noise, but the GAF transformation helps the CNN identify some patterns.

#### Step 3b - MLP on Log-Returns with GAF

The GAF images are flattened into a 576-feature vector and fed to an MLP (Dense(256) → Dropout → Dense(128) → Dropout → Dense(64) → Dense(1)). Using stationary log-returns improves performance. The removal of trend effects allows the model to focus on directional patterns rather than being influenced by price levels.

#### Step 3c - CNN on Fractionally Differenced with GAF

This representation often achieves the best balance, combining stationarity with preserved memory. The CNN captures both short-term directional patterns and longer-term dependencies.

#### Step 3d - CNN Evaluation

GAF-CNN models are computationally more intensive than MLPs but can capture complex spatial patterns invisible to flat-vector models. Stationary inputs consistently improve model reliability. The image-based approach is particularly advantageous for classification tasks.

---

### Step 4 - CNN vs MLP Architecture Comparison

#### How Each Architecture Processes Data

**MLP** receives a flat vector of inputs — either lagged time series values or flattened GAF images. All features are treated independently through fully connected layers. The model has no inherent awareness of sequential order or spatial structure. It is fast, simple, and well-suited for regression on stationary inputs.

**CNN** receives 2D GAF-encoded images and applies convolutional filters that scan local regions of the image, detecting spatial patterns such as angular shapes and textures that correspond to recurring temporal structures. Pooling layers progressively abstract these patterns into higher-level features.

#### Key Observations

1. MLP on regression tasks performs well on stationary inputs but learns the trend rather than return dynamics when given raw price levels, producing misleadingly high R² values.
2. MLP on GAF images loses spatial relationships when images are flattened, making it less efficient than CNN for image-based classification.
3. CNN on GAF images is naturally suited to 2D inputs and detects local angular patterns that represent directional transitions in the time series.
4. Stationary inputs consistently produce more reliable results across both architectures.
5. Fractionally differenced inputs capture long-term dependencies better than log-returns at longer forecasting horizons.

#### Conclusion

CNN with GAF encoding is better suited for direction classification tasks because the image representation exposes geometric patterns that a flat-vector MLP cannot efficiently detect. MLP remains competitive for continuous regression forecasting where the input structure is already informative as a vector. A production system would benefit from ensembling both architectures.

---

## Key Findings

- Training MLP on non-stationary price levels produces misleadingly high R² - the model learns the trend, not the dynamics
- Log-returns provide the most stable and generalizable regression results
- Fractional differencing balances stationarity and memory, often outperforming log-returns at longer forecasting horizons
- GAF-CNN models capture spatial patterns invisible to flat vector MLP models
- Stationary inputs consistently improve model reliability across both architectures

---

## Tech Stack

```
Python 3.x
TensorFlow / Keras
scikit-learn
yfinance
pyts
statsmodels
pandas
numpy
matplotlib
seaborn
```

---

## Installation

```bash
git clone https://github.com/QuantSingularity/CNN-MLP-GAF-Deep-Learning-Finance.git
cd CNN-MLP-GAF-Deep-Learning-Finance
pip install yfinance pyts tensorflow scikit-learn statsmodels matplotlib seaborn pandas numpy
jupyter notebook BTC-MLP-CNN-GAF-Forecasting.ipynb
```

---

## Data

BTC-USD daily closing prices downloaded via `yfinance`:

- **Ticker:** BTC-USD
- **Period:** 2018-01-01 to 2023-06-01
- **Observations:** 2,000 (trimmed)
- **Source:** Yahoo Finance

No manual data download required - the notebook fetches data automatically.

---

## Topics Covered

`deep-learning` `time-series` `bitcoin` `financial-forecasting` `convolutional-neural-network` `mlp` `gramian-angular-field` `fractional-differencing` `stationarity` `adf-test` `log-returns` `quantitative-finance` `tensorflow` `python` `jupyter-notebook`

---

## References

- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_. Wiley.
- Wang, Z., & Oates, T. (2015). Imaging Time-Series to Improve Classification and Imputation. _IJCAI_.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). _Deep Learning_. MIT Press.
