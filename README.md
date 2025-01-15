# Intelligent Trading Model

This repository demonstrates how to train a reinforcement learning (RL) agent using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for trading on hourly candlestick data. The project consists of:

- A custom Gymnasium environment (**environment.py**) that simulates trading with:
  - Multi-bar holding (positions can stay open across multiple time steps).
  - Stop-Loss and Take-Profit checks at each bar.
  - Commission and slippage modeling.
  - Partial position closing (to support scaling out of positions).
  - Detailed logging of each trade for analysis.

- A training script (**train_rl.py**) that:
  - Loads real historical data from a CSV file (`historical_data_1h.csv`).
  - Creates and wraps the environment.
  - Trains a PPO agent.
  - Runs a quick test and prints final results.

---

## Features

- **Historical Data Analysis**: Utilizes technical indicators such as ATR, RSI, and SMA.
- **Custom Loss Function**: Balances classification errors, parameter penalties, and balance rewards.
- **Trainable Parameters**: Automatically adjusts thresholds, risks, and profit/loss ratios.
- **Simulated Trading**: Conducts virtual trades to assess model performance.
- **Detailed Reports**: Logs every decision and trade for analysis.

---

## How It Works

### 1. **Configuration**
The model parameters are defined in `config.json`:
- `window_size`: Number of data points in a sequence.
- `num_features`: Features per data point (e.g., `high`, `low`, `close`, etc.).
- `model_name`: File name for saving and loading the model.
- `data_file`: CSV file containing historical data.
- `initial_balance`, `trade_risk`, `min_balance`, etc.: Parameters for simulated trading.

### 2. **Data Preparation**
The model processes historical market data:
- **Calculate Indicators**: Computes ATR, RSI, and SMAs.
- **Normalization**: Scales data to the [0, 1] range for efficient model training.
- **Sequence Slicing**: Creates overlapping windows of historical data for input to the model.

### 3. **Model Architecture**
The model consists of:
- **Conv1D**: Extracts local patterns in sequences.
- **LSTM**: Captures temporal dependencies.
- **Dense Layer**: Combines features for final prediction.
- **Trainable Parameters**:
  - `long_threshold` and `short_threshold`: Define trading signals.
  - `trade_risk`, `take_profit_ratio`, `stop_loss_ratio`: Optimize trading decisions.

### 4. **Training**
- **Custom Loss Function**: Combines classification loss, parameter regularization, and balance rewards.
- **Simulation**: Predicts market movements, adjusts balances, and logs trades.
- **Dynamic Updates**: Adapts trading parameters during training for market conditions.

### 5. **Reports**
- **Detailed Report**: Logs individual trades, balances, thresholds, and outcomes.
- **Summary Report**: Tracks overall performance metrics, such as success rates and total trades.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/intelligent-trading-model.git
   cd intelligent-trading-model
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare the Configuration**: Edit `config.json` to specify your desired parameters.
4. **Prepare Historical Data**: Ensure you have a CSV file (`historical_data_1h.csv`) with the required columns: `open`, `high`, `low`, `close`, `volume`.

---

## Usage

### Run Training
```bash
python train_rl.py
```

- If `ppo_hourly_model.zip` is found, training continues from the saved weights.
- Otherwise, a new model is created and trained.

### Monitor Logs
- Check training progress in the console or via TensorBoard:
  ```bash
  tensorboard --logdir ./tensorboard_logs/
  ```

### Analyze Results
- Evaluate the model's decisions and performance using the generated reports (`detailed_report_file` and `report_file`).

---

## Output

- **Trained Model**: Saved to `ppo_hourly_model.zip` for future use.
- **Reports**:
  - **Detailed Report**: Logs individual trade metrics, including timestamps, thresholds, positions, and balances.
  - **Summary Report**: Aggregates metrics like total trades, success rates, and final balance.

---

## Example Workflow

1. Collect hourly historical data for a trading pair (e.g., BTC/USDT) using an API or CSV file.
2. Configure the model with `window_size = 168` and `num_features = 5`.
3. Train the model using `train_rl.py`.
4. Analyze trading performance using the detailed and summary reports.

---

## Key Advantages

- **Automation**: End-to-end pipeline from data ingestion to trading simulation.
- **Dynamic Learning**: Adapts thresholds and parameters for evolving market conditions.
- **Comprehensive Reporting**: Provides deep insights into trading strategy effectiveness.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Author

Developed by Ktyby21. Feel free to contact me for suggestions or collaboration opportunities.
