import asyncio
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict
from datetime import datetime, timedelta
import time

class DataProcessor(ABC):
    @abstractmethod
    async def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        pass

class EMAProcessor(DataProcessor):
    def __init__(self, short_span: int, long_span: int):
        self.short_span = short_span
        self.long_span = long_span

    async def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        processed_data = raw_data.copy()
        processed_data['short_ema'] = raw_data['price'].ewm(span=self.short_span).mean()
        processed_data['long_ema'] = raw_data['price'].ewm(span=self.long_span).mean()
        return processed_data

class SignalGenerator(ABC):
    @abstractmethod
    def generate_signal(self, processed_data: pd.DataFrame) -> str:
        pass

class EMASignalGenerator(SignalGenerator):
    def generate_signal(self, processed_data: pd.DataFrame) -> str:
        if processed_data['short_ema'].iloc[-1] > processed_data['long_ema'].iloc[-1]:
            return 'buy'
        elif processed_data['short_ema'].iloc[-1] < processed_data['long_ema'].iloc[-1]:
            return 'sell'
        else:
            return 'hold'

class PortfolioConstructor:
    def __init__(self, signal_generators: List[SignalGenerator], weights: List[float]):
        self.signal_generators = signal_generators
        self.weights = weights

    def generate_portfolio_signal(self, processed_data: pd.DataFrame) -> str:
        signals = [generator.generate_signal(processed_data) for generator in self.signal_generators]
        signal_counts = {'buy': 0, 'sell': 0, 'hold': 0}
        for signal, weight in zip(signals, self.weights):
            signal_counts[signal] += weight
        return max(signal_counts, key=signal_counts.get)

class RiskManager:
    def __init__(self, max_drawdown: float, max_risk_per_trade: float):
        self.max_drawdown = max_drawdown
        self.max_risk_per_trade = max_risk_per_trade

    def calculate_position_size(self, signal: str, price: float, equity: float) -> float:
        if signal == 'buy':
            risk_amount = equity * self.max_risk_per_trade
            position_size = risk_amount / price
        else:
            position_size = 0
        return position_size

    def check_drawdown(self, equity: float, initial_capital: float) -> bool:
        drawdown = (equity - initial_capital) / initial_capital
        return drawdown < -self.max_drawdown

class PerformanceTracker:
    def __init__(self):
        self.equity_history: List[float] = []
        self.trade_history: List[Dict[str, float]] = []

    def update_equity(self, equity: float):
        self.equity_history.append(equity)

    def record_trade(self, trade: Dict[str, float]):
        self.trade_history.append(trade)

    def calculate_performance_metrics(self) -> Dict[str, float]:
        returns = np.diff(self.equity_history) / self.equity_history[:-1]
        cumulative_return = (self.equity_history[-1] - self.equity_history[0]) / self.equity_history[0]
        sharpe_ratio = np.mean(returns) / np.std(returns)
        max_drawdown = np.min(np.cumsum(returns) - np.maximum.accumulate(np.cumsum(returns)))

        return {
            'cumulative_return': cumulative_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

class ExecutionHandler:
    async def execute_trade(self, trade: Dict[str, float]):
        # Implement your trade execution logic here
        # This can include sending orders to a broker or exchange API
        print(f"Executing trade: {trade}")
        await asyncio.sleep(1)  # Simulating trade execution delay

class TradingSystem:
    def __init__(
        self,
        data_processor: DataProcessor,
        portfolio_constructor: PortfolioConstructor,
        risk_manager: RiskManager,
        performance_tracker: PerformanceTracker,
        execution_handler: ExecutionHandler,
        initial_capital: float
    ):
        self.data_processor = data_processor
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager
        self.performance_tracker = performance_tracker
        self.execution_handler = execution_handler
        self.equity = initial_capital

    async def run(self, data_feed):
        async for raw_data in data_feed:
            processed_data = await self.data_processor.process_data(raw_data)
            signal = self.portfolio_constructor.generate_portfolio_signal(processed_data)
            price = raw_data['price'].iloc[-1]

            position_size = self.risk_manager.calculate_position_size(signal, price, self.equity)
            trade = {
                'signal': signal,
                'price': price,
                'size': position_size,
                'timestamp': datetime.now()
            }

            if position_size > 0:
                await self.execution_handler.execute_trade(trade)
                self.performance_tracker.record_trade(trade)

                if signal == 'buy':
                    self.equity -= position_size * price
                elif signal == 'sell':
                    self.equity += position_size * price

            self.performance_tracker.update_equity(self.equity)

            if self.risk_manager.check_drawdown(self.equity, initial_capital):
                print("Max drawdown exceeded. Stopping trading.")
                break

            await asyncio.sleep(1)  # Simulate a delay between each iteration

        performance_metrics = self.performance_tracker.calculate_performance_metrics()
        print(f"Performance Metrics: {performance_metrics}")

async def main():
    # Initialize system components
    data_processor = EMAProcessor(short_span=50, long_span=100)
    signal_generator1 = EMASignalGenerator()
    signal_generator2 = EMASignalGenerator()  # Placeholder for another signal generator
    portfolio_constructor = PortfolioConstructor(
        signal_generators=[signal_generator1, signal_generator2],
        weights=[0.6, 0.4]
    )
    risk_manager = RiskManager(max_drawdown=0.1, max_risk_per_trade=0.02)
    performance_tracker = PerformanceTracker()
    execution_handler = ExecutionHandler()

    trading_system = TradingSystem(
        data_processor=data_processor,
        portfolio_constructor=portfolio_constructor,
        risk_manager=risk_manager,
        performance_tracker=performance_tracker,
        execution_handler=execution_handler,
        initial_capital=100000
    )

    # Simulated real-time data feed
    async def data_feed():
        for _ in range(1000):
            await asyncio.sleep(1)
            yield pd.DataFrame({
                'price': np.random.uniform(50, 150, size=1),
                'volume': np.random.uniform(1000, 10000, size=1)
            })

    await trading_system.run(data_feed())

if __name__ == '__main__':
    asyncio.run(main())