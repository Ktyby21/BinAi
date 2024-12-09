import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv1D, LSTM, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, load_model
import os
import json
import csv
from datetime import datetime
import random

# Load configuration from file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Assign configuration values
window_size = config["window_size"]
num_features = config["num_features"]
normalization_template_file = config["normalization_template_file"]
model_name = config["model_name"]
report_file = config["report_file"]
detailed_report_file = config["detailed_report_file"]
batch_size = config["batch_size"]
initial_balance_range = config["initial_balance"]
trade_risk = config["trade_risk"]
min_balance = config["min_balance"]
neutral_penalty = config["neutral_penalty"]
save_interval = config["save_interval"]
target_balance = config["target_balance"]
data_file = config["data_file"]

# Logging detailed report
def log_detailed_report(timestamp, step, probability, long_threshold, short_threshold, trade_risk, take_profit_ratio, stop_loss_ratio, position, balance, current_price, atr_value, take_profit, stop_loss, exit_price, trade_amount, profit_or_loss):
    """Logs detailed trading decisions into a CSV file."""
    file_exists = os.path.exists(detailed_report_file)
    with open(detailed_report_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Step", "Probability", "Long Threshold", "Short Threshold",
                "Trade Risk", "Take Profit Ratio", "Stop Loss Ratio", "Position",
                "Balance", "Current Price", "ATR Value", "Take Profit",
                "Stop Loss", "Exit Price", "Trade Amount", "Profit/Loss"
            ])
        writer.writerow([
            timestamp, step, round(probability, 3), round(long_threshold, 3), round(short_threshold, 3),
            round(trade_risk, 3), round(take_profit_ratio, 3), round(stop_loss_ratio, 3), position,
            round(balance, 3), round(current_price, 3), round(atr_value, 3),
            round(take_profit, 3) if take_profit else None,
            round(stop_loss, 3) if stop_loss else None,
            round(exit_price, 3) if exit_price else None,
            round(trade_amount, 3), round(profit_or_loss, 3)
        ])

# Indicator calculation
def calculate_indicators(df):
    """Calculates ATR, RSI, and moving averages for the given DataFrame."""
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift()))
    )
    df['atr'] = df['tr'].rolling(14).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    return df

# Load and normalize data
df = pd.read_csv(data_file)
df = calculate_indicators(df)
df.dropna(inplace=True)

if os.path.exists(normalization_template_file):
    with open(normalization_template_file, 'r') as f:
        normalization_template = json.load(f)
else:
    normalization_template = {
        'price_min': df[['high', 'low', 'close']].min().min(),
        'price_max': df[['high', 'low', 'close']].max().max(),
        'volume_min': df['volume'].min(),
        'volume_max': df['volume'].max(),
        'rsi_min': df['rsi'].min(),
        'rsi_max': df['rsi'].max(),
        'sma_min': df[['sma_20', 'sma_50']].min().min(),
        'sma_max': df[['sma_20', 'sma_50']].max().max()
    }
    with open(normalization_template_file, 'w') as f:
        json.dump(normalization_template, f)

def normalize_data(df, template):
    """Normalizes the DataFrame using the given template."""
    df['high'] = (df['high'] - template['price_min']) / (template['price_max'] - template['price_min'])
    df['low'] = (df['low'] - template['price_min']) / (template['price_max'] - template['price_min'])
    df['close'] = (df['close'] - template['price_min']) / (template['price_max'] - template['price_min'])
    df['volume'] = (df['volume'] - template['volume_min']) / (template['volume_max'] - template['volume_min'])
    df['rsi'] = (df['rsi'] - template['rsi_min']) / (template['rsi_max'] - template['rsi_min'])
    df['sma_20'] = (df['sma_20'] - template['sma_min']) / (template['sma_max'] - template['sma_min'])
    df['sma_50'] = (df['sma_50'] - template['sma_min']) / (template['sma_max'] - template['sma_min'])
    return df

df = normalize_data(df, normalization_template)

# Prepare training data
def prepare_training_data(df):
    """Prepares training data by slicing the DataFrame into sequences."""
    X, y = [], []
    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size]
        if window.isnull().values.any():
            continue
        X.append(window[['high', 'low', 'close', 'volume', 'rsi', 'sma_20', 'sma_50']].values)
        y.append(1 if df.iloc[i + window_size]['close'] > df.iloc[i + window_size - 1]['close'] else 0)
    return np.array(X), np.array(y)

X_train, y_train = prepare_training_data(df)
# Model with trainable thresholds
class TradingModel(Model):
    def __init__(self, window_size, num_features):
        super(TradingModel, self).__init__()
        self.input_layer = InputLayer(input_shape=(window_size, num_features))
        self.conv1d = Conv1D(16, kernel_size=2, activation='relu', kernel_regularizer=l2(0.001))
        self.lstm = LSTM(20, return_sequences=False, kernel_regularizer=l2(0.001))
        self.dense = Dense(16, activation='relu', kernel_regularizer=l2(0.001))
        self.output_layer = Dense(1, activation='sigmoid')

        # Trainable parameters
        self.long_threshold = tf.Variable(0.51, trainable=True, dtype=tf.float32, name='long_threshold')
        self.short_threshold = tf.Variable(0.49, trainable=True, dtype=tf.float32, name='short_threshold')
        self.trade_risk = tf.Variable(0.02, trainable=True, dtype=tf.float32, name='trade_risk')
        self.take_profit_ratio = tf.Variable(1.5, trainable=True, dtype=tf.float32, name='take_profit_ratio')
        self.stop_loss_ratio = tf.Variable(1.0, trainable=True, dtype=tf.float32, name='stop_loss_ratio')

    def call(self, inputs, training=False):
        """Defines the forward pass of the model."""
        x = self.input_layer(inputs)
        x = self.conv1d(x)
        x = self.lstm(x)
        x = self.dense(x)
        probability = self.output_layer(x)
        
        # Enforcing constraints on thresholds and probabilities
        delta = 0.1
        self.long_threshold.assign(tf.clip_by_value(self.long_threshold, 0.0 + delta, 1.0))
        self.short_threshold.assign(tf.clip_by_value(self.short_threshold, 0.0, 1.0 - delta))

        # Ensuring Long Threshold >= Short Threshold + delta
        self.long_threshold.assign(tf.maximum(self.long_threshold, self.short_threshold + delta))
        self.short_threshold.assign(tf.minimum(self.short_threshold, self.long_threshold - delta))
        
        # Clipping trainable parameters within predefined limits
        self.trade_risk.assign(tf.clip_by_value(self.trade_risk, 0.01, 1.0))
        self.take_profit_ratio.assign(tf.maximum(self.take_profit_ratio, 0.1))
        self.stop_loss_ratio.assign(tf.maximum(self.stop_loss_ratio, 0.1))
        
        # Clipping probability within a valid range
        probability = tf.clip_by_value(probability, 0.01, 0.99)
        
        return probability, self.long_threshold, self.short_threshold, self.trade_risk, self.take_profit_ratio, self.stop_loss_ratio

def custom_loss(y_true, y_pred_tuple, balance):
    """Defines the custom loss function for the model."""
    probability, long_threshold, short_threshold, trade_risk, take_profit_ratio, stop_loss_ratio = y_pred_tuple

    # Reshape y_true to match probability dimensions
    y_true = tf.cast(tf.reshape(y_true, tf.shape(probability)), dtype=tf.float32)

    # Cross-entropy loss penalizing incorrect predictions
    classification_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, probability))

    # Penalty for threshold proximity or extreme values
    threshold_penalty = tf.reduce_mean(tf.square(long_threshold - short_threshold - 0.1))

    # Encourage decision confidence based on thresholds
    decision_quality_penalty = tf.reduce_mean(
        tf.square(probability - tf.clip_by_value(long_threshold, 0.4, 0.6)) +
        tf.square(probability - tf.clip_by_value(short_threshold, 0.3, 0.5))
    )

    # Reward for maintaining balance close to the target
    balance_reward = tf.cast(tf.maximum(balance / target_balance, 0.0), dtype=tf.float32)

    # Regularization for trainable parameters
    regularization = (
        tf.reduce_sum(tf.square(long_threshold)) +
        tf.reduce_sum(tf.square(short_threshold)) +
        tf.reduce_sum(tf.square(trade_risk)) +
        tf.reduce_sum(tf.square(take_profit_ratio)) +
        tf.reduce_sum(tf.square(stop_loss_ratio))
    )

    # Total loss combining all components
    loss = (
        classification_loss
        + 0.1 * threshold_penalty
        + 0.1 * decision_quality_penalty
        - 5 * balance_reward
        + 0.001 * regularization
    )

    return loss

def normalize_window(df_window):
    """
    Normalizes the given DataFrame window based on its min/max values.
    """
    df_window = df_window.copy()  # Prevent modifications to the original DataFrame
    min_values = df_window.min()
    max_values = df_window.max()
    normalized_window = (df_window - min_values) / (max_values - min_values)
    return normalized_window, min_values, max_values

# Logging training report
def log_report(timestamp, step, balance, successful_trades, total_trades, loss):
    """Logs summary of training metrics to a CSV file."""
    success_rate = (successful_trades / total_trades) * 100 if total_trades > 0 else 0
    if isinstance(loss, tf.Tensor):
        loss = loss.numpy()
    with open(report_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, step, balance, successful_trades, success_rate, loss])

# Training or loading the model
if os.path.exists(model_name):
    model = tf.keras.models.load_model(model_name, custom_objects={'TradingModel': TradingModel})
    print("Existing model loaded for training.")
else:
    model = TradingModel(window_size, num_features)
    print("New model initialized.")

optimizer = tf.keras.optimizers.legacy.Adam()
epoch = 1
start_time = datetime.now()
steps, successful_trades, total_trades = 0, 0, 0

# Main training loop
for epoch in range(10):
    balance = random.uniform(900, 1100)
    for i in range(0, len(X_train) - batch_size, batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss = custom_loss(y_batch, y_pred, balance)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Simulate trading decisions and update balance
        for j in range(len(X_batch)):
            steps += 1
            position = None
            current_price = 0.0
            atr_value = 0.0
            take_profit = None
            stop_loss = None
            trade_amount = 0.0
            profit_or_loss = 0.0
            previous_balance = balance
            probability, long_threshold, short_threshold, trade_risk, take_profit_ratio, stop_loss_ratio = model(X_batch[j].reshape(1, window_size, num_features))
            probability = probability[0][0].numpy()
            trade_risk = trade_risk.numpy()
            take_profit_ratio = take_profit_ratio.numpy()
            stop_loss_ratio = stop_loss_ratio.numpy()
            
            neutral_gap = 0.005

            if short_threshold + neutral_gap < probability < long_threshold - neutral_gap:
                position = None
            else:
                distance_to_long = abs(probability - long_threshold)
                distance_to_short = abs(probability - short_threshold)
                position = "long" if distance_to_long < distance_to_short else "short"

            if position is None:
                balance -= balance * neutral_penalty 

            current_price = df['close'].iloc[window_size + i + j]
            atr_value = df['atr'].iloc[window_size + i + j]

            take_profit = current_price + (atr_value * take_profit_ratio) if position == "long" else current_price - (atr_value * take_profit_ratio)
            stop_loss = current_price - (atr_value * stop_loss_ratio) if position == "long" else current_price + (atr_value * stop_loss_ratio)

            current_high = df['high'].iloc[window_size + i + j]
            current_low = df['low'].iloc[window_size + i + j]

            exit_price = df['close'].iloc[window_size + i + j + 1]
            trade_amount = balance * trade_risk
            trade_executed = False

            if position:
                profit_or_loss = (
                    trade_amount * ((exit_price - current_price) / current_price)
                    if position == "long"
                    else trade_amount * ((current_price - exit_price) / current_price)
                )
                balance += profit_or_loss

            if position == "long":
                if current_low <= stop_loss:
                    balance -= trade_amount
                    trade_executed = True
                elif current_high >= take_profit:
                    profit = trade_amount * (take_profit_ratio - 1.0)
                    balance += profit
                    successful_trades += 1
                    trade_executed = True
                else:
                    profit_or_loss = trade_amount * ((exit_price - current_price) / current_price)
                    balance += profit_or_loss
                    if profit_or_loss > 0:
                        successful_trades += 1
                    trade_executed = True

            elif position == "short":
                if current_high >= stop_loss:
                    balance -= trade_amount
                    trade_executed = True
                elif current_low <= take_profit:
                    profit = trade_amount * (take_profit_ratio - 1.0)
                    balance += profit
                    successful_trades += 1
                    trade_executed = True
                else:
                    profit_or_loss = trade_amount * ((current_price - exit_price) / current_price)
                    balance += profit_or_loss
                    if profit_or_loss > 0:
                        successful_trades += 1
                    trade_executed = True
            
            log_detailed_report(
                timestamp=datetime.now(),
                step=steps,
                probability=probability,
                long_threshold=long_threshold.numpy(),
                short_threshold=short_threshold.numpy(),
                trade_risk=trade_risk,
                take_profit_ratio=take_profit_ratio,
                stop_loss_ratio=stop_loss_ratio,
                position=position,
                balance=balance,
                current_price=current_price,
                atr_value=atr_value,
                take_profit=take_profit if position else None,
                stop_loss=stop_loss if position else None,
                exit_price=exit_price,
                trade_amount=trade_amount,
                profit_or_loss=profit_or_loss
            )
            if trade_executed:
                total_trades += 1

            if steps % 500 == 0:
                model.trade_risk.assign_add(tf.random.uniform([], 0.002, 0.03))
            if steps % save_interval == 0:
                log_report(datetime.now(), steps, balance, successful_trades, total_trades, loss)
                print(f"Step {steps}: Report successfully logged.")
            if balance <= min_balance:
                print(f"Balance dropped below minimum. Reset triggered at step {steps}.")
                balance = random.uniform(900, 1100)
                steps, successful_trades, total_trades =  0, 0, 0
                model.save(model_name)
                j = 0
                break
    steps, successful_trades, total_trades =  0, 0, 0
    model.save(model_name)
    print(f"Epoch {epoch}: Model saved successfully.")