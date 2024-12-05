### Ethereum Price Prediction using LSTM
This project leverages a Long Short-Term Memory (LSTM) deep learning model to predict Ethereum (ETH) prices using historical market data. It preprocesses time-series data, trains the model, and evaluates its predictive accuracy, providing a basis for actionable insights in the volatile cryptocurrency market.

### Features
**Data Preprocessing:** Handles missing values, normalizes data, and creates sequences for time-series analysis.
**Deep Learning:** Implements an LSTM-based neural network for price prediction.
**Visualization:** Provides visual comparisons of predicted vs. actual prices and tracks training/validation loss over epochs.
**Model Saving:** Includes functionality to save and load trained models for future predictions.

### Technologies Used
The project relies on the following technologies:
- **Python**: Core programming language.
- **TensorFlow/Keras**: Deep learning framework for building the LSTM model.
- **Pandas and NumPy**: Data manipulation libraries.
- **Matplotlib**: Visualization of results and training progress.
- **Scikit-Learn**: Data normalization with MinMaxScaler.

### Installation
**1. Clone the repository:**
`git clone https://github.com/lleahhhh/eth_price_prediction.git
cd eth_price_prediction`

**2. Install the required packages:**
`pip install -r requirements.txt`
Ensure you have Python 3.7+ and TensorFlow 2.0+ installed.

**3. Add the historical Ethereum price dataset:**
- Add `ETHUSD_1m_Combined_Index.csv` to the project directory.

### Data Description
The dataset consists of historical Ethereum price data with the following columns:
`Open time`: Timestamp of the data entry.
`Open`, `High`, `Low`, `Close`: Prices during the time interval.
`Volume`: Traded volume during the time interval.
The project focuses on the `Close` price for prediction.

### Workflow
**1. Data Pre-processing:**
- Convert timestamps to datetime format and set as the DataFrame index.
- Normalize the `Close` price using MinMaxScaler.
- Create sequences of 60 historical prices for training.

**2. Model Development:**
- Build an LSTM model with two LSTM layers and Dense layers for price prediction.
- Train the model using an 80/20 train-test split.

**3. Evaluation:**
- Evaluate the model’s performance on the test set using loss metrics.
- Visualize predicted vs. actual prices.

**4. Model Saving and Loading:**
- Save the trained model for future predictions using TensorFlow's .h5 format.
- Load the model to ensure compatibility.

### Results
**Training vs. Validation Loss:** 
- Monitored over 5 epochs to prevent overfitting.
**Price Prediction Accuracy:**
- Sample of predictions compared against actual prices for validation.
- Visualization demonstrates alignment between predicted and actual Ethereum prices.

### Usage
- Run the script:
`python eth_price_prediction.py`
- The model will process the data, train the LSTM model, evaluate its performance, and display results.

### Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes. Thanks for making it this far!
