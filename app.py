import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MarketData:
    price: float
    market_pressure: float
    volume_skew: float
    timestamp: datetime

class OptionsDFT:
    def __init__(
        self, 
        symbol: str, 
        check_interval_minutes: int = 5, 
        decay_rate: float = 0.9, 
        threshold: float = 0.5,
        lookback_periods: int = 5
    ):
        """
        Initialize the Options DFT analyzer.
        
        Args:
            symbol: Stock ticker symbol
            check_interval_minutes: Time between market checks in minutes
            decay_rate: Rate at which old signals decay (0 to 1)
            threshold: Threshold for generating buy/sell signals
            lookback_periods: Number of periods to average for final decision
        """
        self.symbol = symbol.upper()
        self.check_interval_minutes = check_interval_minutes
        self.decay_rate = self._validate_decay_rate(decay_rate)
        self.threshold = threshold
        self.lookback_periods = lookback_periods
        self.ticker = yf.Ticker(symbol)
        
        # Initialize market hours
        self._init_market_hours()
        
    @staticmethod
    def _validate_decay_rate(decay_rate: float) -> float:
        """Validate decay rate is between 0 and 1"""
        if not 0 <= decay_rate <= 1:
            raise ValueError("Decay rate must be between 0 and 1")
        return decay_rate
        
    def _init_market_hours(self) -> None:
        """Initialize market hours in Eastern Time"""
        et_tz = pytz.timezone('US/Eastern')
        today = datetime.now(et_tz)
        
        self.market_open = today.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        self.market_close = today.replace(
            hour=16, minute=0, second=0, microsecond=0
        )
        
        trading_minutes = int((16 - 9.5) * 60)
        self.num_checks = trading_minutes // self.check_interval_minutes
        
    def _get_nearest_expiry(self) -> str:
        """Get the nearest options expiration date"""
        expiration_dates = self.ticker.options
        if not expiration_dates:
            raise ValueError(f"No options data available for {self.symbol}")
        return min(expiration_dates)
        
    def fetch_options_data(self) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
        """
        Fetch current price and options chain data.
        
        Returns:
            Tuple of (current_price, calls_df, puts_df)
        """
        try:
            current_price = float(
                self.ticker.history(period='1d', interval='1m')['Close'].iloc[-1]
            )
            
            options = self.ticker.option_chain(self._get_nearest_expiry())
            return current_price, options.calls, options.puts
            
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {str(e)}")
            
    def calculate_market_pressure(
        self, 
        calls: pd.DataFrame, 
        puts: pd.DataFrame, 
        current_price: float
    ) -> float:
        """
        Calculate market pressure using volume-weighted distance from current price.
        
        Returns:
            Normalized pressure between -1 (bearish) and 1 (bullish)
        """
        def calculate_pressure(df: pd.DataFrame) -> float:
            volume = df['volume'].fillna(0)
            distance = np.exp(-np.abs(df['strike'] - current_price) / current_price)
            return float((volume * distance).sum())
            
        call_pressure = calculate_pressure(calls)
        put_pressure = calculate_pressure(puts)
        
        total_pressure = call_pressure + put_pressure
        if total_pressure == 0:
            return 0.0
            
        return float(np.clip((call_pressure - put_pressure) / total_pressure, -1, 1))
        
    def calculate_volume_profile(
        self, 
        calls: pd.DataFrame, 
        puts: pd.DataFrame, 
        current_price: float
    ) -> float:
        """
        Calculate volume profile skew using volume-weighted average prices.
        
        Returns:
            Skew value indicating bias in option volumes
        """
        def calculate_vwap(df: pd.DataFrame) -> float:
            volume = df['volume'].fillna(0)
            if volume.sum() == 0:
                return current_price
            return float(np.average(df['strike'], weights=volume))
            
        call_vwap = calculate_vwap(calls)
        put_vwap = calculate_vwap(puts)
        
        call_skew = (call_vwap - current_price) / current_price
        put_skew = (put_vwap - current_price) / current_price
        
        return float(call_skew - put_skew)
        
    def run_dft(self) -> Tuple[np.ndarray, List[datetime], List[MarketData]]:
        """
        Run DFT analysis collecting time series data.
        
        Returns:
            Tuple of (decision_values, timestamps, market_data)
        """
        times: List[datetime] = []
        decision_values: List[float] = []
        market_data: List[MarketData] = []
        
        attractiveness = 0.0
        direction_preference = 0.0
        
        for check in range(self.num_checks):
            current_time = self.market_open + timedelta(
                minutes=check * self.check_interval_minutes
            )
            
            # Skip if outside market hours
            if current_time > self.market_close:
                continue
                
            # Fetch and process data
            try:
                current_price, calls, puts = self.fetch_options_data()
                
                market_pressure = self.calculate_market_pressure(
                    calls, puts, current_price
                )
                volume_skew = self.calculate_volume_profile(
                    calls, puts, current_price
                )
                
                # Update DFT components
                combined_signal = (market_pressure + volume_skew) / 2
                direction_preference = float(
                    (1 - self.decay_rate) * (combined_signal - attractiveness) + 
                    self.decay_rate * direction_preference
                )
                attractiveness += direction_preference
                
                # Store results
                times.append(current_time)
                decision_values.append(float(attractiveness))
                market_data.append(MarketData(
                    price=current_price,
                    market_pressure=market_pressure,
                    volume_skew=volume_skew,
                    timestamp=current_time
                ))
                
            except Exception as e:
                print(f"Error during check {check}: {str(e)}")
                continue
                
        return np.array(decision_values), times, market_data
        
    def get_signal(self, decision_values: np.ndarray) -> Tuple[str, float]:
        """Generate trading signal based on recent decision values"""
        final_decision = float(np.mean(
            decision_values[-self.lookback_periods:]
        ))
        
        if final_decision > self.threshold:
            signal = "BULLISH"
        elif final_decision < -self.threshold:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
            
        return signal, final_decision
        
    def plot_analysis(
        self,
        decision_values: np.ndarray,
        times: List[datetime],
        market_data: List[MarketData]
    ) -> None:
        """Create visualization of analysis results"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot decision values
        ax1.plot(times, decision_values, label='DFT Decision Values', color='blue')
        ax1.axhline(y=self.threshold, color='g', linestyle='--', label='Bullish Threshold')
        ax1.axhline(y=-self.threshold, color='r', linestyle='--', label='Bearish Threshold')
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Decision Value')
        ax1.set_title(f'{self.symbol} DFT Analysis')
        ax1.grid(True)
        ax1.legend()
        
        # Plot market pressure and volume skew
        pressures = [d.market_pressure for d in market_data]
        skews = [d.volume_skew for d in market_data]
        ax2.plot(times, pressures, label='Market Pressure', color='purple')
        ax2.plot(times, skews, label='Volume Skew', color='orange')
        ax2.set_xlabel('Time (ET)')
        ax2.set_ylabel('Signal Strength')
        ax2.grid(True)
        ax2.legend()
        
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
    def print_analysis_data(
        self,
        decision_values: np.ndarray,
        times: List[datetime],
        market_data: List[MarketData]
    ) -> None:
        """Print detailed analysis data in tabular format"""
        print("\nDetailed Analysis Data:")
        print("-" * 100)
        print(f"{'Time (ET)':12} {'Price':>10} {'Decision':>10} {'Market Pressure':>15} {'Volume Skew':>12}")
        print("-" * 100)
        
        for i in range(len(times)):
            print(f"{times[i].strftime('%H:%M:%S'):12} "
                  f"${market_data[i].price:>9.2f} "
                  f"{decision_values[i]:>10.3f} "
                  f"{market_data[i].market_pressure:>15.3f} "
                  f"{market_data[i].volume_skew:>12.3f}")
        
        print("-" * 100)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Average Decision Value: {np.mean(decision_values):.3f}")
        print(f"Max Decision Value: {np.max(decision_values):.3f}")
        print(f"Min Decision Value: {np.min(decision_values):.3f}")
        print(f"Standard Deviation: {np.std(decision_values):.3f}")
        print(f"Price Change: ${market_data[-1].price - market_data[0].price:.2f}")
        print(f"Price Change %: {((market_data[-1].price / market_data[0].price) - 1) * 100:.2f}%")
    
    def analyze_and_plot(self) -> None:
        """Run complete analysis and display results"""
        decision_values, times, market_data = self.run_dft()
        
        if not decision_values.size or not market_data:
            print(f"No data collected for {self.symbol}")
            return
            
        signal, final_decision = self.get_signal(decision_values)
        last_data = market_data[-1]
        
        # Print the data table
        self.print_analysis_data(decision_values, times, market_data)
        
        # Plot results
        self.plot_analysis(decision_values, times, market_data)
        
        # Print analysis summary
        print(f"\nAnalysis Results for {self.symbol}:")
        print(f"Current Price: ${last_data.price:.2f}")
        print(f"Signal: {signal}")
        print(f"Decision Strength: {final_decision:.3f}")
        print(f"Final Market Pressure: {last_data.market_pressure:.3f}")
        print(f"Final Volume Skew: {last_data.volume_skew:.3f}")
        print(f"Time Period: {times[0].strftime('%H:%M')} - {times[-1].strftime('%H:%M')} ET")


def run_analysis(symbol: str = "SPY", interval_minutes: int = 5) -> None:
    """Run DFT analysis with error handling"""
    try:
        dft = OptionsDFT(symbol, check_interval_minutes=interval_minutes)
        dft.analyze_and_plot()
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        raise

if __name__ == "__main__":
    run_analysis("SPY")