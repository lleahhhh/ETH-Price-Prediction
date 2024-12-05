# Error handling
try:

    #  ----- Step 1.0: import data manipulation libraries ----- 
    import pandas as pd 
    import numpy as np
    import matplotlib as plt

    # import data reprocessing libraries
    from sklearn.preprocessing import MinMaxScaler

    # import Lont Short-Term Memory Deep Learning Model (LSTM) with TensorFlow/Keras 
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, LSTM

    # check if libraries are imported by printing their versions
    print("TensorFlow version:", tf.__version__)
    print("Pandas version:", pd.__version__)
    print("NumPy version:", np.__version__)
    print("Matplotlib version:", plt.__version__)
    print("Scikit-Learn version:", MinMaxScaler.__module__.split('.')[0])

    # Confirm TensorFlow is using GPU if available 
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("Using GPU:", physical_devices[0])
    else:
        print("GPU not available, using CPU.")

    # Load the CAS file into the Pandas Dataframe
    df = pd.read_csv("C:\\Users\\leahl\\Downloads\\ETHUSD_1m_Combined_Index.csv")

    # Step 1.1: Explore the Ethereum price data
    print(df.head())
    print(df.info())

    # Timeseries data needs to be handled correctly - convert "Open Time" to datetime format and set as the index
    df['Open time'] = pd.to_datetime(df['Open time'])
    df.set_index('Open time', inplace=True)

    # Check for missing values
    print("Missing values in each column:\n", df.isnull().sum())

    # Drop unnecessary data columns - don't need 'Volume' for price prediction
    df = df[['Open', 'High', 'Low', 'Close']]

    # Display the first few rows to verify changes
    print(df.head())
    print(df.info())

    # ----- Step 2: Data Preprocessing for LSTM Model ----
    # Step 2.1: Normalise the data - scale values of the columns to a range between 0-1 for better model performance.
    # Use MinMaxScaler to normalise the 'Close' column - the column of prediction focus.
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_close = scaler.fit_transform(df[['Close']])

    # Print first 5 normalized values to check
    print("Sample of normalized data:\n", normalized_close[:5])  

    # Step 2.2 Prepare Sequences for Training and Testing
    # Define sequence length 
    sequence_length = 60

    # Create sequences and corresponding targets
    # Initialize 2 lists:
        # sequences: store each sequence of historical prices.
        # targets: store the target ("next" price) that we want the model to predict, based on each sequence.
    sequences = []
    targets = []

    # Initialise a loop iterating over the normalized closing data to create sequences of a specified length (e.g. 60), each paired with a target value (the "next" value after each sequence)    
    for i in range(len(normalized_close) - sequence_length):
        
    # Sequence of `sequence_length` data points
        sequences.append(normalized_close[i:i + sequence_length])

    # Determines target value for each sequence - aka next value immediately following each sequence:
        targets.append(normalized_close[i + sequence_length])

    # Convert sequences and targets to numpy arrays for training
    sequences = np.array(sequences)
    targets = np.array(targets)

    # Check "shape" of sequences - targets NumPy arrays - helps verify that data is in the correct format for training a LSTM models.
    print("Shape of sequences:", sequences.shape)
    print("Shape of targets:", targets.shape)

    # ----- Step 3: Split Data into Training & Testing Sets (Typical split = 80% training, 20% testing) ----
    # Define the training & testing split ratio
    split_ratio = 0.8
    split_index = int(len(sequences) * split_ratio)

    # Split the data
    x_train, x_test = sequences[:split_index], sequences[split_index:] # X_train contains 80% of sequences for training the model. X_test contains 20% of sequences.
    y_train, y_test = targets[:split_index], targets[split_index:]

    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    # ----- Step 4: Build the LSTM Model ----

    # Define the LSTM Model 
    model = Sequential()
    # Purpose: Creates a Sequential model, which is a linear stack of layers. A standard way of building models in Keras when each layer has one input and one output.
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1))) # 1st LSTM layer w 50 units
    # Purpose: Adds the first LSTM layer to the model with 50 units (neurons), specifies that the layer will return sequences, and sets the input shape.
   
    model.add(LSTM(50, return_sequences=False))  # 2nd LSTM layer
    # Purpose: Adds a second LSTM layer with 50 units and return_sequences=False.
    
    model.add(Dense(25))  # Dense layer w 25 units
    # Theory: Dense Layer: This is a fully connected layer, meaning each neuron is connected to every neuron in the previous layer. 
        # Theory: Dense layers are typically used to process the information extracted by the LSTM layers and prepare it for the output.
    model.add(Dense(1))   # Output layer w 1 unit for price prediction
   
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("LSTM Model architecture created successfully.")

    # ----- Step 5: Train the LSTM Model ----
    # Train the model
    epochs = 5  # Epoch = one full pass through entire training dataset. 
    batch_size = 64  # Number of samples model processes before updating its weights. Instead of updating weights after every single sample, use batches for more efficieny. 
        # Model processes 64 training samples at a time before updating its internal weights. Smaller batch sizes result in more freq. weight updates - may improve learning but increase computation time. 

    # Add early stopping to prevent overfitting 
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3) # Stop training if validation loss doesn't improve for 3 epochs

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[early_stopping])
    # Purpose: Train LSTM model using training data and evaluate it during training using the validation (testing) data.

    # Plot training vs. validation loss
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate model on the test set
    loss = model.evaluate(x_test, y_test)
        # Purpose: Evaluate the trained model on the test set (data not used during training) to measure its performance.
    print("Test loss:", loss)

    # ----- Step 6: Make Preidctions on the Test Data ----
    # Make predictions on the test set
    predictions = model.predict(x_test)
        # model.predict(x_test) - uses trained model to predict next closing price based on test set (x_test)
        # Each row in x_test is a sequence of normalised closing prices, the model predicts next price in seq.
        # Output is a NumPy array where each element corresponds to the predicted next price for seq. in x_test.

    # Save the model
    model.save("eth_price_prediction_model.h5")
    print("Ethereum price prediction model saved successfully.")

    # Test loading the saved model
    from tensorflow.keras.models import load_model
    saved_model = load_model("eth_price_prediction_model.h5")
    print("Ethereum price prediction model loaded successfully.")


    # Undo the scaling on the predictions and actual test values
    predictions = scaler.inverse_transform(predictions)
        # Purpose: Convert normalised predictions back to the original price scale for easier interpretation.
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        # Purpose: Convert actual test target values (y_test) back to original price scale for comparison w predictions.

    # Display sample predictions
    print("Sample Predictions vs Actuals:")
    for i in range(5):
        print(f"Predicted: {predictions[i][0]}, Actual: {y_test_unscaled[i][0]}")
        # Purpose: Display predicted vs. actual prices for first 5 samples in the test set for a quick comparison.

    # ----- Step 7: Plot the Results ----
    # Plotting prediction vs. actual values
    plt.figure(figsize=(12,6))
    plt.plot(y_test_unscaled, color='#ff69b4', label='Actual Prices') # Hot pink colour
    plt.plot(predictions, color='#b469ff', label='Predicted Prices') # Light purple colour
    plt.title('Ethereum Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($USD)')
    plt.legend()
    plt.show()

# Error handling
except Exception as e:
    print(f"Error encountered: {e}")