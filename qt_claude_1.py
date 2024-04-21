import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Dict, List

class DataProcessor(ABC):
    @abstractmethod
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        pass

class MovingAverageProcessor(DataProcessor):
    def __init__(self, span: int):
        self.span = span

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        processed_data = raw_data.copy()
        processed_data['buy_ema'] = processed_data['buy_volume'].ewm(span=self.span).mean()
        processed_data['sell_ema'] = processed_data['sell_volume'].ewm(span=self.span).mean()
        return processed_data

class QuantTradingSystem:
    def __init__(self, data_processor: DataProcessor, risk_parameters: Dict[str, float]):
        self.data_processor = data_processor
        self.risk_parameters = risk_parameters
        self.trades: List[Dict[str, float]] = []
        self.current_positions: Dict[str, int] = {}

    def ingest_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Ingest and preprocess raw data."""
        # Perform data validation and cleaning
        cleaned_data = self._validate_and_clean_data(raw_data)
        # Process the cleaned data
        processed_data = self.data_processor.process_data(cleaned_data)
        return processed_data

    def _validate_and_clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean raw data."""
        # Perform data validation checks
        if raw_data.isnull().values.any():
            raise ValueError("Input data contains NaN values.")
        # Perform data cleaning
        cleaned_data = raw_data.dropna()
        return cleaned_data

    def generate_signals(self, processed_data: pd.DataFrame) -> str:
        """Generate trading signals based on processed data."""
        latest_data = processed_data.iloc[-1]
        if latest_data['buy_ema'] > latest_data['sell_ema']:
            return 'buy'
        elif latest_data['sell_ema'] > latest_data['buy_ema']:
            return 'sell'
        else:
            return 'hold'

    def execute_trade(self, signal: str, processed_data: pd.DataFrame):
        """Execute trades based on the signal."""
        timestamp = datetime.now()
        price = processed_data['price'].iloc[-1]
        if signal == 'buy':
            # Implement your buy order logic here
            trade_info = {
                'timestamp': timestamp,
                'type': 'buy',
                'price': price,
                'size': self._calculate_position_size(price)
            }
            self._send_order(trade_info)
        elif signal == 'sell':
            # Implement your sell order logic here
            trade_info = {
                'timestamp': timestamp,
                'type': 'sell',
                'price': price,
                'size': self._calculate_position_size(price)
            }
            self._send_order(trade_info)
        else:
            print(f"No trade executed at {timestamp}")

    def _calculate_position_size(self, price: float) -> int:
        """Calculate the position size based on risk management rules."""
        # Implement your position sizing logic here
        # Example: Fixed percentage of equity
        equity = self._calculate_equity()
        risk_percentage = self.risk_parameters['risk_per_trade']
        position_size = int((equity * risk_percentage) / price)
        return position_size

    def _send_order(self, trade_info: Dict[str, float]):
        """Send the order to the execution platform."""
        # Implement your order sending logic here
        self.trades.append(trade_info)
        self._update_positions(trade_info)
        print(f"Executed {trade_info['type']} trade of {trade_info['size']} shares at {trade_info['price']} on {trade_info['timestamp']}")

    def _update_positions(self, trade: Dict[str, float]):
        """Update current positions based on executed trades."""
        if trade['type'] == 'buy':
            self.current_positions['stock'] = self.current_positions.get('stock', 0) + trade['size']
        elif trade['type'] == 'sell':
            self.current_positions['stock'] = self.current_positions.get('stock', 0) - trade['size']

    def _calculate_equity(self) -> float:
        """Calculate the current equity of the trading account."""
        # Implement your equity calculation logic here
        # Example: Sum of cash and current position values
        cash = self.risk_parameters['initial_capital']
        position_value = self.current_positions.get('stock', 0) * self.trades[-1]['price']
        equity = cash + position_value
        return equity

    def manage_risk(self, processed_data: pd.DataFrame):
        """Apply risk management rules to mitigate losses."""
        equity = self._calculate_equity()
        if equity < self.risk_parameters['max_drawdown'] * self.risk_parameters['initial_capital']:
            print("Max drawdown exceeded, liquidating positions.")
            self.execute_trade('sell', processed_data)

    def run(self, raw_data: pd.DataFrame):
        """Run the trading strategy."""
        processed_data = self.ingest_data(raw_data)
        signal = self.generate_signals(processed_data)
        self.execute_trade(signal, processed_data)
        self.manage_risk(processed_data)

# Example usage
if __name__ == "__main__":
    # Define risk parameters
    risk_params = {
        'initial_capital': 100000,
        'max_drawdown': 0.1,
        'risk_per_trade': 0.02
    }

    # Create an instance of the data processor
    data_processor = MovingAverageProcessor(span=12)

    # Create an instance of the trading system
    trading_system = QuantTradingSystem(data_processor=data_processor, risk_parameters=risk_params)

    # Example raw data
    raw_data = pd.DataFrame({
        'buy_volume': np.random.rand(100) * 100,
        'sell_volume': np.random.rand(100) * 100,
        'price': np.random.rand(100) * 100 + 100
    })

    # Run the trading system
    trading_system.run(raw_data)