#!/usr/bin/env python3
"""
live_trading.py
Production-ready live trading bot for Groww API
Run with: python live_trading.py --config config.json
"""
import argparse
import json
import time
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import joblib
import numpy as np
import pandas as pd
import torch

from models.lstm import LSTMClassifier
from utils.features import FeatureEngine
from utils.risk import RiskManager
from utils.portfolio import PortfolioTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GrowwAPI:
    """Wrapper for Groww API calls with error handling and rate limiting"""
    
    def __init__(self, access_token: str, api_key: str):
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "X-API-VERSION": "1.0"
        }
        self.base_url = "https://api.groww.in/v1"
        self.last_call_time = 0
        self.min_call_interval = 0.1  # 100ms between calls
        
    def _rate_limit(self):
        """Simple rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_quote(self, instrument_token: str, segment: str = "CASH") -> Dict:
        """Get live quote for instrument"""
        self._rate_limit()
        url = f"{self.base_url}/live-data/quote"
        params = {"instrument_token": instrument_token, "segment": segment}
        
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching quote for {instrument_token}: {e}")
            raise
    
    def get_ohlc(self, instrument_token: str, interval: str = "5m", 
                 start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Dict:
        """Get OHLC candles for instrument"""
        self._rate_limit()
        url = f"{self.base_url}/historical/candle/range"
        params = {
            "instrument_token": instrument_token,
            "interval": interval
        }
        if start_ts:
            params["start"] = start_ts
        if end_ts:
            params["end"] = end_ts
        
        try:
            resp = requests.get(url, headers=self.headers, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching OHLC for {instrument_token}: {e}")
            raise
    
    def place_order(self, order_payload: Dict) -> Dict:
        """Place order on Groww"""
        self._rate_limit()
        url = f"{self.base_url}/order/create"
        
        try:
            resp = requests.post(url, headers=self.headers, json=order_payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"Error placing order: {e}")
            raise


class SignalGenerator:
    """Generate trading signals using ensemble of models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load RandomForest
        self.rf_model = joblib.load(config['rf_model_path'])
        self.rf_scaler = joblib.load(config['rf_scaler_path'])
        
        # Load LSTM if configured
        self.lstm_model = None
        if config.get('lstm_model_path'):
            lstm_features = joblib.load(config['lstm_features_path'])
            self.lstm_model = LSTMClassifier(
                input_size=len(lstm_features),
                hidden_size=config.get('lstm_hidden_size', 64),
                num_layers=config.get('lstm_num_layers', 2)
            ).to(self.device)
            self.lstm_model.load_state_dict(
                torch.load(config['lstm_model_path'], map_location=self.device)
            )
            self.lstm_model.eval()
        
        self.feature_engine = FeatureEngine()
        self.seq_len = config.get('sequence_length', 20)
        
    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        """Generate trading signal from dataframe"""
        if len(df) < self.seq_len:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.seq_len}")
            return None
        
        # Get RF signal
        rf_signal = self._rf_predict(df)
        
        # Get LSTM signal if available
        lstm_signal = None
        if self.lstm_model is not None:
            lstm_signal = self._lstm_predict(df)
        
        # Ensemble logic
        if rf_signal and lstm_signal:
            # Both models must agree for high confidence
            if rf_signal['pred'] == lstm_signal['pred']:
                confidence = (rf_signal['prob_up'] + lstm_signal['prob_up']) / 2
                return {
                    'pred': rf_signal['pred'],
                    'confidence': confidence,
                    'model': 'ensemble',
                    'price': float(df['close'].iloc[-1])
                }
            else:
                # Disagreement - use higher confidence model
                if rf_signal['prob_up'] > lstm_signal['prob_up']:
                    return {**rf_signal, 'model': 'rf_fallback'}
                else:
                    return {**lstm_signal, 'model': 'lstm_fallback'}
        
        # Fallback to available model
        return rf_signal or lstm_signal
    
    def _rf_predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get RandomForest prediction"""
        features = ["returns", "volatility", "ma5", "ma20", "rsi"]
        row = df[features].tail(1)
        
        if row.empty or row.isnull().any().any():
            return None
        
        X = self.rf_scaler.transform(row.values)
        pred = self.rf_model.predict(X)[0]
        prob = self.rf_model.predict_proba(X)[0]
        
        return {
            'pred': int(pred),
            'prob_up': float(prob[1]),
            'prob_down': float(prob[0]),
            'confidence': float(max(prob)),
            'price': float(df['close'].iloc[-1]),
            'model': 'rf'
        }
    
    def _lstm_predict(self, df: pd.DataFrame) -> Optional[Dict]:
        """Get LSTM prediction"""
        lstm_features = joblib.load(self.config['lstm_features_path'])
        
        if len(df) < self.seq_len:
            return None
        
        seq = df[lstm_features].tail(self.seq_len).values
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.lstm_model(x)
            prob = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred = int(np.argmax(prob))
        
        return {
            'pred': pred,
            'prob_up': float(prob[1]),
            'prob_down': float(prob[0]),
            'confidence': float(max(prob)),
            'price': float(df['close'].iloc[-1]),
            'model': 'lstm'
        }


class LiveTrader:
    """Main trading orchestrator"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.api = GrowwAPI(
            self.config['access_token'],
            self.config['api_key']
        )
        self.signal_gen = SignalGenerator(self.config)
        self.feature_engine = FeatureEngine()
        self.risk_manager = RiskManager(self.config)
        self.portfolio = PortfolioTracker()
        
        self.watchlist = self.config['watchlist']
        self.symbol_to_token = self.config['symbol_to_token']
        self.execute_orders = self.config.get('execute_orders', False)
        self.interval = self.config.get('interval', '5m')
        self.interval_seconds = self.config.get('interval_seconds', 60)
        
        self.ledger_path = Path(self.config.get('ledger_path', 'ledger.json'))
        self.ledger = self._load_ledger()
        
        logger.info(f"Initialized LiveTrader - Paper: {not self.execute_orders}")
        logger.info(f"Watching {len(self.watchlist)} symbols")
    
    def _load_config(self, path: str) -> Dict:
        """Load and validate configuration"""
        with open(path) as fh:
            config = json.load(fh)
        
        required = ['access_token', 'watchlist', 'symbol_to_token', 
                   'rf_model_path', 'rf_scaler_path']
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        return config
    
    def _load_ledger(self) -> List[Dict]:
        """Load existing ledger if present"""
        if self.ledger_path.exists():
            with open(self.ledger_path) as fh:
                return json.load(fh)
        return []
    
    def _save_ledger(self):
        """Save ledger to disk"""
        with open(self.ledger_path, 'w') as fh:
            json.dump(self.ledger, fh, indent=2)
    
    def fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process data for a symbol"""
        try:
            token = self.symbol_to_token[symbol]
            
            # Fetch last 2 days of data
            end_ts = int(time.time() * 1000)
            start_ts = int((datetime.utcnow() - timedelta(days=2)).timestamp() * 1000)
            
            raw = self.api.get_ohlc(token, self.interval, start_ts, end_ts)
            candles = raw.get('candles') or raw.get('data') or raw
            
            if not candles:
                logger.warning(f"No candle data for {symbol}")
                return None
            
            # Parse candles
            if isinstance(candles, list) and isinstance(candles[0], list):
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            else:
                df = pd.DataFrame(candles)
            
            # Add features
            df = self.feature_engine.compute(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def process_signal(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Process signal through risk management and execute if approved"""
        
        # Check confidence threshold
        min_confidence = self.config.get('min_confidence', 0.6)
        if signal['confidence'] < min_confidence:
            logger.info(f"{symbol}: Low confidence {signal['confidence']:.3f}, skipping")
            return None
        
        # Risk check
        if not self.risk_manager.check_trade(symbol, signal, self.portfolio):
            logger.info(f"{symbol}: Trade rejected by risk manager")
            return None
        
        # Determine position size
        position_size = self.risk_manager.calculate_position_size(
            symbol, signal, self.portfolio
        )
        
        if position_size == 0:
            logger.info(f"{symbol}: Position size = 0, skipping")
            return None
        
        # Prepare order
        side = "BUY" if signal['pred'] == 1 else "SELL"
        order = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'price': signal['price'],
            'size': position_size,
            'confidence': signal['confidence'],
            'model': signal['model']
        }
        
        # Execute or simulate
        if self.execute_orders:
            result = self._place_live_order(symbol, order)
        else:
            result = self._place_paper_order(symbol, order)
        
        return result
    
    def _place_paper_order(self, symbol: str, order: Dict) -> Dict:
        """Simulate order placement"""
        order['status'] = 'paper_filled'
        self.ledger.append(order)
        self.portfolio.update_position(order)
        logger.info(f"PAPER ORDER: {symbol} {order['side']} {order['size']} @ {order['price']:.2f}")
        return order
    
    def _place_live_order(self, symbol: str, order: Dict) -> Dict:
        """Place actual order via Groww API"""
        payload = {
            "instrument_token": self.symbol_to_token[symbol],
            "exchange": "NSE",
            "transaction_type": order['side'],
            "quantity": order['size'],
            "order_type": "MARKET",
            "product": "INTRADAY"
        }
        
        try:
            response = self.api.place_order(payload)
            order['status'] = 'live_placed'
            order['order_id'] = response.get('order_id')
            order['response'] = response
            self.ledger.append(order)
            self.portfolio.update_position(order)
            logger.info(f"LIVE ORDER: {symbol} {order['side']} {order['size']} @ {order['price']:.2f} - ID: {order.get('order_id')}")
            return order
        except Exception as e:
            logger.error(f"Failed to place order for {symbol}: {e}")
            order['status'] = 'failed'
            order['error'] = str(e)
            return order
    
    def run(self):
        """Main trading loop"""
        logger.info("=" * 60)
        logger.info("Starting live trading loop")
        logger.info(f"Mode: {'LIVE' if self.execute_orders else 'PAPER'}")
        logger.info(f"Interval: {self.interval_seconds}s")
        logger.info("=" * 60)
        
        try:
            iteration = 0
            while True:
                iteration += 1
                loop_start = time.time()
                logger.info(f"\n--- Iteration {iteration} ---")
                
                for symbol in self.watchlist:
                    try:
                        # Fetch data
                        df = self.fetch_data(symbol)
                        if df is None or len(df) == 0:
                            continue
                        
                        # Generate signal
                        signal = self.signal_gen.generate_signal(df, symbol)
                        if signal is None:
                            continue
                        
                        logger.info(f"{symbol}: {signal['model']} pred={signal['pred']} conf={signal['confidence']:.3f}")
                        
                        # Process signal
                        result = self.process_signal(symbol, signal)
                        
                    except Exception as e:
                        logger.exception(f"Error processing {symbol}: {e}")
                        continue
                
                # Save ledger after each iteration
                self._save_ledger()
                
                # Portfolio summary
                summary = self.portfolio.get_summary()
                logger.info(f"Portfolio: {summary['num_positions']} positions, PnL: ₹{summary['total_pnl']:.2f}")
                
                # Sleep until next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.interval_seconds - elapsed)
                logger.info(f"Iteration complete in {elapsed:.1f}s, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("\n" + "=" * 60)
            logger.info("Trading loop stopped by user")
            logger.info("=" * 60)
            self._save_ledger()
            self.portfolio.print_summary()
            logger.info(f"Ledger saved to: {self.ledger_path}")


def main():
    parser = argparse.ArgumentParser(description='Live Trading Bot for Groww')
    parser.add_argument('--config', required=True, help='Path to config.json')
    args = parser.parse_args()
    
    trader = LiveTrader(args.config)
    trader.run()


if __name__ == "__main__":
    main()