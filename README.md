# Intelligent Trading Model

This repository demonstrates how to use Reinforcement Learning (with Stable Baselines3 and Gym) for automated trading on hourly candlestick data. The project consists of:

  - A custom Gym environment (environment.py) that simulates trading with:
        Multi-bar holding (positions can stay open across multiple time steps).
        Built-in Stop-Loss and Take-Profit checks inside each candle.
        Commission and slippage modeling.
        Partial position closing (to support scaling out of positions).
        Logging each trade for later analysis.

   - A training script (train_rl.py) that:
        Loads real historical data from a CSV file (historical_data_1h.csv).
        Creates and wraps the environment.
        Trains a PPO agent.
        Runs a quick test and prints final results.

---

## Features

- **Historical Data Analysis**: Utilizes technical indicators such as ATR, RSI, and SMA.
- **Custom Loss Function**: Includes classification errors, parameter penalties, and balance rewards.
- **Trainable Parameters**: Automatically adjusts thresholds, risks, and profit/loss ratios.
- **Simulated Trading**: Conducts virtual trades to assess model performance.
- **Detailed Reports**: Logs every decision and trade for analysis.

---

## How It Works

### 1. **Configuration**
The model parameters are defined in the `config.json` file:
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
- **Detailed Report** (`detailed_report_file`): Logs individual trades, balances, thresholds, and outcomes.
- **Summary Report** (`report_file`): Tracks overall performance metrics, such as success rates and total trades.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/intelligent-trading-model.git
   cd intelligent-trading-model
   ```
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Prepare the Configuration: Edit config.json to specify your desired parameters.
5. Prepare Historical Data: Use a script (e.g., fetch_binance_data.py) to collect market data and save it as a CSV file.

Usage

Run Training:
python train_model.py
Monitor Logs: Check progress in detailed_report_file and report_file.
Analyze Results: Evaluate the model's decisions and performance using the generated reports.

Output

- Reports
- Detailed Report: Captures individual trade metrics, including timestamps, thresholds, positions, and balances.
- Summary Report: Aggregates metrics like total trades, success rates, and final balance.

Model
The trained model is saved to a file (model_name), enabling future reuse or fine-tuning.

Example Workflow

- Collect 4-hour historical data for BNB/USDT using Binance API.
- Configure the model with window_size = 100 and num_features = 7.
- Train the model using the script.
- Analyze trading performance using the reports.

  
Key Advantages

- Automation: End-to-end pipeline from data ingestion to trading simulation.
- Dynamic Learning: Adapts thresholds and parameters for evolving market conditions.
- Comprehensive Reporting: Provides deep insights into trading strategy effectiveness.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Developed by Ktyby21. Feel free to contact me for suggestions or collaboration opportunities.
