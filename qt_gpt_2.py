import numpy as np
import pandas as pd
from datetime import datetime
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Process incoming raw data to a usable form for analysis."""
        pass

class EMACalculator(DataProcessor):
    def __init__(self, span: int):
        self.span = span

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        ema_data = raw_data.ewm(span=self.span).mean()
        return ema_data

class SignalGenerator(ABC):
    @abstractmethod
    def generate_signal(self, processed_data: pd.DataFrame) -> str:
        """Generate a trading signal based on processed data."""
        pass

class EMASignalGenerator(SignalGenerator):
    def generate_signal(self, processed_data: pd.DataFrame) -> str:
        if processed_data['buy_ema'].iloc[-1] > processed_data['sell_ema'].iloc[-1]:
            return 'buy'
        elif processed_data['sell_ema'].iloc[-1] > processed_data['buy_ema'].iloc[-1]:
            return 'sell'
        else:
            return 'hold'

class OrderExecutor:
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.position_size = 0
        self.positions = {}

    def execute_trade(self, signal: str, price: float):
        if signal in ['buy', 'sell']:
            self.position_size = self.calculate_position_size(price)
            self.update_positions(signal, price)
            print(f"Executed {signal} trade at {price}")

    def calculate_position_size(self, price: float) -> int:
        risk_amount = self.capital * 0.01  # Risk 1% of capital per trade
        position_size = int(risk_amount / price)
        return position_size

    def update_positions(self, signal: str, price: float):
        if signal == 'buy':
            self.positions['stock'] = self.positions.get('stock', 0) + self.position_size
            self.capital -= self.position_size * price
        elif signal == 'sell':
            self.positions['stock'] = self.positions.get('stock', 0) - self.position_size
            self.capital += self.position_size * price

class TradingSystem:
    def __init__(self, data_processor: DataProcessor, signal_generator: SignalGenerator, order_executor: OrderExecutor):
        self.data_processor = data_processor
        self.signal_generator = signal_generator
        self.order_executor = order_executor

    def run(self, raw_data: pd.DataFrame):
        processed_data = self.data_processor.process_data(raw_data)
        signal = self.signal_generator.generate_signal(processed_data)
        last_price = raw_data['price'].iloc[-1]
        self.order_executor.execute_trade(signal, last_price)

# Usage Example
if __name__ == "__main__":
    # Initialize system components
    ema_processor = EMACalculator(span=12)
    signal_generator = EMASignalGenerator()
    order_executor = OrderExecutor(initial_capital=100000)

    # Initialize and run the trading system
    trading_system = TradingSystem(data_processor=ema_processor, signal_generator=signal_generator, order_executor=order_executor)

    # Example raw data
    raw_data = pd.DataFrame({
        'buy_volume': np.random.rand(100) * 100,
        'sell_volume': np.random.rand(100) * 100,
        'price': np.random.rand(100) * 150 + 50
    })

    trading_system.run(raw_data)