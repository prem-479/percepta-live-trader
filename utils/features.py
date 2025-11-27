"""
utils/features.py
Feature engineering for trading signals
"""
import pandas as pd
import numpy as np


class FeatureEngine:
    """Compute technical indicators and features"""
    
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all features to dataframe"""
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            if df['timestamp'].dtype != 'datetime64[ns]':
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Sort by time
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype(float)
        
        # Basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=5).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()
        
        # Moving averages
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        # Exponential moving averages
        df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Technical indicators
        df['rsi'] = self._compute_rsi(df['close'], window=14)
        df['macd'], df['macd_signal'] = self._compute_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self._compute_bollinger_bands(df['close'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(float)
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-9)
        
        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # High-Low range
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_range'] = (df['high'] - df['low']) / df['close']
            df['hl_avg'] = (df['high'] + df['low']) / 2
        
        # Target (for training)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN rows
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        """Compute Relative Strength Index"""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _compute_macd(self, series: pd.Series, 
                      fast: int = 12, slow: int = 26, signal: int = 9):
        """Compute MACD and signal line"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        
        return macd, macd_signal
    
    def _compute_bollinger_bands(self, series: pd.Series, 
                                  window: int = 20, num_std: float = 2.0):
        """Compute Bollinger Bands"""
        ma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        
        upper = ma + (num_std * std)
        lower = ma - (num_std * std)
        
        return upper, lower