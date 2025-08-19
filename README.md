# Intelligent Trading Model

This repository demonstrates how to train a reinforcement learning (RL) agent using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for trading on hourly candlestick data. The project consists of:

- A custom Gymnasium environment (**env/hourly_trading_env.py**) that simulates trading with:
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
  - Stores checkpoints, logs and TensorBoard data under `runs/`.

---

## Features

- **ATR-based Stop Loss and Take Profit** for risk management.
- **Partial Position Closing** to scale out of trades.
- **Simulated Trading** with commission and slippage modelling.
- **Detailed Trade Logging** for analysis.

---

## How It Works

### 1. **Configuration**
The model parameters are defined in `config.json`:
- `window_size`: Number of data points in a sequence.
- `model_name`: File name for saving and loading the model.
- `data_file`: CSV file containing historical data.
- `initial_balance`, `trade_risk`, `min_balance`, etc.: Parameters for simulated trading.
- `data_top_n`, `data_interval`, `data_start_date`, `data_end_date`, `data_output_file`: Parameters for downloading market history for the most traded pairs.

### 2. **Data Preparation**
The model processes historical market data:
- **Calculate Indicators**: Computes ATR for each bar.
- **Normalization**: Scales data for efficient training.
- **Sequence Slicing**: Creates windows of historical data for input to the agent.

### 3. **Training**
- Uses the PPO algorithm from Stable-Baselines3.
- Simulates trades, updates balances and logs results.

### 4. **Reports**
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
  tensorboard --logdir ./runs/tensorboard/
  ```

### Analyze Results
- Evaluate the model's decisions and performance using the generated reports (`detailed_report_file` and `report_file`).

---

## Output

- **Trained Model**: Saved to `runs/ppo_hourly_model.zip` for future use.
- **Reports**:
  - **Detailed Report**: Logs individual trade metrics, including timestamps, thresholds, positions, and balances.
  - **Summary Report**: Aggregates metrics like total trades, success rates, and final balance.

---

## Example Workflow

1. Collect hourly historical data for a trading pair (e.g., BTC/USDT) or use `get_data.py` to download top trading pairs from Binance.
2. Configure the model with `window_size = 168`.
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
