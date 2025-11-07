
import MetaTrader5 as mt5

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import os
import asyncio
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from fredapi import Fred
import warnings
from typing import Dict, List, Optional
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import requests

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = "forex_liquidity_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chart settings
plt.style.use('seaborn-v0_8')
DPI = 100
WIDTH_PX = 750
HEIGHT_PX = 500
FIGSIZE = (WIDTH_PX/DPI, HEIGHT_PX/DPI)
plt.rcParams['figure.figsize'] = FIGSIZE
plt.rcParams['figure.dpi'] = DPI

class GlobalLiquidityMiner:
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize Global Liquidity Data Mining Module."""
        self.fred_api_key = fred_api_key
        self.fred = Fred(api_key=fred_api_key) if fred_api_key else None
        self.data_cache = {}
        self.liquidity_indicators = {}
        self.start_date = '2008-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        logger.info("Global Liquidity Miner initialized")

    def fetch_central_bank_balance_sheets(self) -> Dict[str, pd.DataFrame]:
        """Fetch central bank balance sheet data from various sources."""
        balance_sheets = {}
        try:
            # Federal Reserve Balance Sheet (FRED)
            if self.fred:
                logger.info("Fetching Federal Reserve balance sheet data...")
                try:
                    fed_data = self.fred.get_series('WALCL', start=self.start_date, end=self.end_date)
                    balance_sheets['FED'] = pd.DataFrame({
                        'date': fed_data.index,
                        'balance_sheet': fed_data.values,
                        'currency': 'USD'
                    })
                    logger.info(f"Fed data points: {len(fed_data)}")
                except Exception as e:
                    logger.warning(f"Failed to fetch Fed data: {e}")

            # ECB Balance Sheet (Alternative: ECB API or yfinance fallback)
            logger.info("Fetching ECB balance sheet data...")
            try:
                # Try ECB API (simplified example, requires actual API access)
                # Note: ECB data requires registration at https://data.ecb.europa.eu/
                # Fallback to yfinance if API not available
                eur_money_supply = yf.download('EURGBP=X', start=self.start_date, end=self.end_date, progress=False)
                if not eur_money_supply.empty:
                    balance_sheets['ECB'] = pd.DataFrame({
                        'date': eur_money_supply.index,
                        'balance_sheet': eur_money_supply['Close'].values * 1000000,  # Proxy scaling
                        'currency': 'EUR'
                    })
                    logger.info(f"ECB proxy data points: {len(eur_money_supply)}")
                else:
                    logger.warning("ECB yfinance data empty, trying alternative source...")
                    # Placeholder for alternative ECB data source (e.g., manual CSV or other API)
            except Exception as e:
                logger.warning(f"Could not fetch ECB data: {e}")

            # Bank of Japan (BOJ) Balance Sheet (yfinance or alternative)
            logger.info("Fetching BOJ proxy data...")
            try:
                jpy_data = yf.download('USDJPY=X', start=self.start_date, end=self.end_date, progress=False)
                if not jpy_data.empty:
                    balance_sheets['BOJ'] = pd.DataFrame({
                        'date': jpy_data.index,
                        'balance_sheet': (1/jpy_data['Close'].values) * 10000000,  # Proxy scaling
                        'currency': 'JPY'
                    })
                    logger.info(f"BOJ proxy data points: {len(jpy_data)}")
            except Exception as e:
                logger.warning(f"Could not fetch BOJ data: {e}")

            # People's Bank of China (PBOC) Balance Sheet
            logger.info("Fetching PBOC proxy data...")
            try:
                cny_data = yf.download('USDCNY=X', start=self.start_date, end=self.end_date, progress=False)
                if not cny_data.empty:
                    balance_sheets['PBOC'] = pd.DataFrame({
                        'date': cny_data.index,
                        'balance_sheet': cny_data['Close'].values * 1000000,  # Proxy scaling
                        'currency': 'CNY'
                    })
                    logger.info(f"PBOC proxy data points: {len(cny_data)}")
            except Exception as e:
                logger.warning(f"Could not fetch PBOC data: {e}")

        except Exception as e:
            logger.error(f"Error fetching central bank data: {e}")

        self.data_cache['balance_sheets'] = balance_sheets
        return balance_sheets

    def fetch_money_supply_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch money supply indicators."""
        money_supply = {}
        try:
            if self.fred:
                logger.info("Fetching US M2 money supply...")
                try:
                    m2_data = self.fred.get_series('M2SL', start=self.start_date, end=self.end_date)
                    money_supply['US_M2'] = pd.DataFrame({
                        'date': m2_data.index,
                        'value': m2_data.values,
                        'indicator': 'M2_Supply',
                        'country': 'USA'
                    })
                    logger.info(f"US M2 data points: {len(m2_data)}")
                except Exception as e:
                    logger.warning(f"Failed to fetch M2 data: {e}")

                logger.info("Fetching US monetary base...")
                try:
                    base_data = self.fred.get_series('BOGMBASE', start=self.start_date, end=self.end_date)
                    money_supply['US_BASE'] = pd.DataFrame({
                        'date': base_data.index,
                        'value': base_data.values,
                        'indicator': 'Monetary_Base',
                        'country': 'USA'
                    })
                    logger.info(f"US monetary base data points: {len(base_data)}")
                except Exception as e:
                    logger.warning(f"Failed to fetch monetary base data: {e}")

            logger.info("Fetching global liquidity proxies...")
            try:
                tlt_data = yf.download('TLT', start=self.start_date, end=self.end_date, progress=False)
                if not tlt_data.empty:
                    money_supply['BOND_LIQUIDITY'] = pd.DataFrame({
                        'date': tlt_data.index,
                        'value': tlt_data['Close'].values,
                        'indicator': 'Bond_Liquidity',
                        'country': 'Global'
                    })
                    logger.info(f"Bond liquidity data points: {len(tlt_data)}")
            except Exception as e:
                logger.warning(f"Could not fetch TLT data: {e}")

            try:
                vix_data = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
                if not vix_data.empty:
                    money_supply['VIX'] = pd.DataFrame({
                        'date': vix_data.index,
                        'value': vix_data['Close'].values,
                        'indicator': 'Volatility_Index',
                        'country': 'Global'
                    })
                    logger.info(f"VIX data points: {len(vix_data)}")
            except Exception as e:
                logger.warning(f"Could not fetch VIX data: {e}")

        except Exception as e:
            logger.error(f"Error fetching money supply data: {e}")

        self.data_cache['money_supply'] = money_supply
        return money_supply

    def calculate_global_liquidity_index(self) -> pd.DataFrame:
        """Calculate composite global liquidity index."""
        try:
            logger.info("Calculating Global Liquidity Index...")
            all_data = []
            
            if 'balance_sheets' in self.data_cache:
                for bank, df in self.data_cache['balance_sheets'].items():
                    df_normalized = df.copy()
                    df_normalized['normalized'] = (df['balance_sheet'] - df['balance_sheet'].mean()) / df['balance_sheet'].std()
                    df_normalized['source'] = f'{bank}_Balance_Sheet'
                    all_data.append(df_normalized[['date', 'normalized', 'source']])
            
            if 'money_supply' in self.data_cache:
                for indicator, df in self.data_cache['money_supply'].items():
                    df_normalized = df.copy()
                    df_normalized['normalized'] = (df['value'] - df['value'].mean()) / df['value'].std()
                    df_normalized['source'] = indicator
                    all_data.append(df_normalized[['date', 'normalized', 'source']])
            
            if not all_data:
                logger.warning("No data available for liquidity index calculation")
                return pd.DataFrame()
            
            combined_data = pd.concat(all_data, ignore_index=True)
            daily_index = combined_data.groupby('date')['normalized'].mean().reset_index()
            daily_index.columns = ['date', 'liquidity_index']
            
            daily_index['ma_30'] = daily_index['liquidity_index'].rolling(30).mean()
            daily_index['ma_90'] = daily_index['liquidity_index'].rolling(90).mean()
            
            daily_index['liquidity_regime'] = pd.cut(
                daily_index['liquidity_index'], 
                bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
                labels=['Very_Tight', 'Tight', 'Neutral', 'Loose', 'Very_Loose']
            )
            
            self.liquidity_indicators['global_index'] = daily_index
            output_path = os.path.join(OUTPUT_DIR, "global_liquidity_index.csv")
            daily_index.to_csv(output_path, index=False)
            logger.info(f"Global Liquidity Index saved to {output_path}")
            return daily_index
            
        except Exception as e:
            logger.error(f"Error calculating liquidity index: {e}")
            return pd.DataFrame()

class ForexLiquidityForecaster:
    def __init__(self, fred_api_key: Optional[str] = None):
        """Initialize Forex forecasting module with global liquidity data."""
        self.liquidity_miner = GlobalLiquidityMiner(fred_api_key)
        self.data_cache = {}
        self.models = {}
        self.scalers = {}
        self.forecasts = {}
        self.start_date = '2008-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not mt5.initialize():
            logger.error("MetaTrader5 initialization failed")
            raise Exception("MT5 initialization failed")
        logger.info("MetaTrader5 initialized successfully")

    def fetch_forex_data(self, symbol: str, timeframe: int = mt5.TIMEFRAME_D1, days: int = 730) -> pd.DataFrame:
        """Fetch historical Forex data from MetaTrader5."""
        try:
            logger.info(f"Fetching {symbol} data for timeframe {timeframe}...")
            utc_from = datetime.now() - timedelta(days=days)
            rates = mt5.copy_rates_from(symbol, timeframe, utc_from, days)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(rates)
            df['date'] = pd.to_datetime(df['time'], unit='s')
            df = df[['date', 'open', 'high', 'low', 'close', 'tick_volume']]
            df.set_index('date', inplace=True)
            
            df['price_change_1d'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(5)
            df['price_change_20d'] = df['close'].pct_change(20)
            df['volatility'] = df['price_change_1d'].rolling(20).std()
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching {symbol} data: {e}")
            return pd.DataFrame()

    def prepare_features_for_prediction(self, target_symbol: str) -> pd.DataFrame:
        """Prepare features for ML prediction model using global liquidity data."""
        try:
            logger.info(f"Preparing features for {target_symbol} prediction...")
            
            target_data = self.fetch_forex_data(target_symbol)
            if target_data.empty:
                logger.warning(f"No data for {target_symbol}")
                return pd.DataFrame()

            target_data['target_price_change'] = target_data['close'].shift(-1) / target_data['close'] - 1
            
            feature_df = target_data[['close', 'tick_volume', 'price_change_1d', 'price_change_5d', 
                                     'price_change_20d', 'volatility', 'volume_sma', 'target_price_change']].copy()
            
            self.liquidity_miner.fetch_central_bank_balance_sheets()
            self.liquidity_miner.fetch_money_supply_data()
            self.liquidity_miner.calculate_global_liquidity_index()
            
            if 'global_index' in self.liquidity_miner.liquidity_indicators:
                liquidity_df = self.liquidity_miner.liquidity_indicators['global_index'].set_index('date')
                feature_df = feature_df.join(liquidity_df[['liquidity_index', 'ma_30', 'ma_90']], how='left')
                feature_df['liquidity_change'] = feature_df['liquidity_index'].pct_change()
                feature_df['liquidity_momentum'] = feature_df['liquidity_index'] - feature_df['ma_30']
                feature_df['liquidity_trend'] = feature_df['ma_30'] - feature_df['ma_90']
            
            if 'balance_sheets' in self.liquidity_miner.data_cache:
                for bank, df in self.liquidity_miner.data_cache['balance_sheets'].items():
                    bank_df = df.set_index('date')[['balance_sheet']].rename(columns={'balance_sheet': f'{bank}_balance'})
                    feature_df = feature_df.join(bank_df, how='left')
                    feature_df[f'{bank}_balance'] = feature_df[f'{bank}_balance'].fillna(method='ffill')
                    feature_df[f'{bank}_balance_change'] = feature_df[f'{bank}_balance'].pct_change()
            
            if 'money_supply' in self.liquidity_miner.data_cache:
                for indicator, df in self.liquidity_miner.data_cache['money_supply'].items():
                    supply_df = df.set_index('date')[['value']].rename(columns={'value': indicator})
                    feature_df = feature_df.join(supply_df, how='left')
                    feature_df[indicator] = feature_df[indicator].fillna(method='ffill')
                    feature_df[f'{indicator}_change'] = feature_df[indicator].pct_change()
            
            for lag in [1, 5, 10]:
                feature_df[f'price_change_lag_{lag}'] = feature_df['price_change_1d'].shift(lag)
                feature_df[f'volatility_lag_{lag}'] = feature_df['volatility'].shift(lag)
            
            feature_df = feature_df.dropna()
            
            logger.info(f"Prepared {len(feature_df)} samples with {feature_df.shape[1]-1} features for {target_symbol}")
            return feature_df
            
        except Exception as e:
            logger.error(f"Error preparing features for {target_symbol}: {e}")
            return pd.DataFrame()

    def train_prediction_model(self, target_symbol: str):
        """Train ML model to predict Forex price movements."""
        try:
            logger.info(f"Training prediction model for {target_symbol}...")
            
            feature_df = self.prepare_features_for_prediction(target_symbol)
            if feature_df.empty:
                logger.warning(f"No data available for training {target_symbol} model")
                return
            
            X = feature_df.drop(columns=['target_price_change'])
            y = feature_df['target_price_change']
            
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            
            logger.info(f"Model for {target_symbol}:")
            logger.info(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
            logger.info(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            
            self.models[target_symbol] = model
            self.scalers[target_symbol] = scaler
            
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_path = os.path.join(OUTPUT_DIR, f"feature_importance_{target_symbol}.csv")
            feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Feature importance saved to {importance_path}")
            
            self.models[f'{target_symbol}_metrics'] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'features': list(X.columns),
                'n_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Error training model for {target_symbol}: {e}")

    async def get_price_forecast(self, target_symbol: str, forecast_days: int = 5) -> dict:
        """Generate price forecasts for target Forex pair."""
        try:
            if target_symbol not in self.models:
                logger.warning(f"No trained model available for {target_symbol}")
                return {'symbol': target_symbol, 'forecasts': [], 'confidence': 0.0}
            
            logger.info(f"Generating {forecast_days}-day forecast for {target_symbol}...")
            
            feature_df = self.prepare_features_for_prediction(target_symbol)
            if feature_df.empty:
                logger.warning(f"No data available for forecasting {target_symbol}")
                return {'symbol': target_symbol, 'forecasts': [], 'confidence': 0.0}
            
            model = self.models[target_symbol]
            scaler = self.scalers[target_symbol]
            
            current_price = feature_df['close'].iloc[-1]
            forecasts = []
            
            latest_features = feature_df.drop(columns=['target_price_change']).iloc[-1:].copy()
            
            for day in range(1, forecast_days + 1):
                X_scaled = scaler.transform(latest_features)
                price_change_pred = model.predict(X_scaled)[0]
                
                if day == 1:
                    forecast_price = current_price * (1 + price_change_pred)
                else:
                    forecast_price = forecasts[-1]['price'] * (1 + price_change_pred)
                
                metrics = self.models.get(f'{target_symbol}_metrics', {})
                confidence = max(0.0, min(1.0, metrics.get('test_r2', 0.5)))
                
                forecasts.append({
                    'day': day,
                    'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'price': forecast_price,
                    'price_change': price_change_pred,
                    'price_change_pct': price_change_pred * 100,
                    'confidence': confidence
                })
                
                if day < forecast_days:
                    latest_features.iloc[0, latest_features.columns.get_loc('price_change_1d')] = price_change_pred
                    latest_features.iloc[0, latest_features.columns.get_loc('close')] = forecast_price
            
            self.forecasts[target_symbol] = {
                'current_price': current_price,
                'forecasts': forecasts,
                'generated_at': datetime.now().isoformat(),
                'model_confidence': confidence
            }
            
            forecast_path = os.path.join(OUTPUT_DIR, f"forecast_{target_symbol}.json")
            with open(forecast_path, 'w') as f:
                json.dump(self.forecasts[target_symbol], f, indent=2)
            
            logger.info(f"Forecast for {target_symbol} saved to {forecast_path}")
            
            return {
                'symbol': target_symbol,
                'current_price': current_price,
                'forecasts': forecasts,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast for {target_symbol}: {e}")
            return {'symbol': target_symbol, 'forecasts': [], 'confidence': 0.0}

    def visualize_price_forecasts(self, target_symbol: str):
        """Visualize price forecasts for target Forex pair."""
        if target_symbol not in self.forecasts:
            logger.warning(f"No forecasts available for {target_symbol}")
            return
        
        try:
            forecast_data = self.forecasts[target_symbol]
            historical_data = self.fetch_forex_data(target_symbol).tail(30)
            
            fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
            
            ax.plot(historical_data.index, historical_data['close'], 
                   linewidth=2, color='#1f77b4', alpha=0.8, label='Historical Prices')
            
            forecast_dates = [datetime.strptime(f['date'], '%Y-%m-%d') for f in forecast_data['forecasts']]
            forecast_prices = [f['price'] for f in forecast_data['forecasts']]
            
            connection_dates = [historical_data.index[-1]] + forecast_dates
            connection_prices = [historical_data['close'].iloc[-1]] + forecast_prices
            
            ax.plot(connection_dates, connection_prices, 
                   linewidth=2, color='#ff7f0e', alpha=0.8, label='Forecast', linestyle='--')
            
            confidence = forecast_data['model_confidence']
            upper_bound = [p * (1 + (1-confidence) * 0.1) for p in forecast_prices]
            lower_bound = [p * (1 - (1-confidence) * 0.1) for p in forecast_prices]
            
            ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                           alpha=0.3, color='orange', label=f'Confidence Band ({confidence:.1%})')
            
            ax.axhline(y=forecast_data['current_price'], color='red', 
                      linestyle=':', alpha=0.7, label='Current Price')
            
            ax.set_title(f'{target_symbol} Price Forecast', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, f"forecast_{target_symbol}_chart.png")
            plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
            plt.close()
            logger.info(f"Forecast chart for {target_symbol} saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating forecast visualization for {target_symbol}: {e}")

    async def run_full_analysis_with_predictions(self, target_symbols: list = None):
        """Run complete Forex analysis with liquidity-based predictions."""
        logger.info("Starting Forex Analysis with Liquidity-Based Predictions...")
        
        if target_symbols is None:
            target_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        await asyncio.sleep(0.1)
        self.liquidity_miner.fetch_central_bank_balance_sheets()
        await asyncio.sleep(0.1)
        self.liquidity_miner.fetch_money_supply_data()
        await asyncio.sleep(0.1)
        self.liquidity_miner.calculate_global_liquidity_index()
        
        for symbol in target_symbols:
            self.train_prediction_model(symbol)
            await self.get_price_forecast(symbol, forecast_days=5)
            self.visualize_price_forecasts(symbol)
            await asyncio.sleep(0.1)
        
        self.liquidity_miner.visualize_central_bank_balance_sheets()
        self.liquidity_miner.visualize_global_liquidity_index()
        
        logger.info("Forex Analysis with Liquidity-Based Predictions completed successfully!")

if __name__ == "__main__":
    try:
        # Prompt for FRED API key
        fred_api_key = input("Enter FRED API key (or press Enter to skip): ").strip() or None
        forecaster = ForexLiquidityForecaster(fred_api_key=fred_api_key)
        target_symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        asyncio.run(forecaster.run_full_analysis_with_predictions(target_symbols))
        logger.info("=== FOREX ANALYSIS COMPLETE ===")
        logger.info("Generated files:")
        logger.info("1. forecast_[SYMBOL]_chart.png - Individual symbol price forecasts")
        logger.info("2. feature_importance_[SYMBOL].csv - ML feature importance")
        logger.info("3. forecast_[SYMBOL].json - ML forecast data")
        logger.info("4. central_bank_balance_sheets.png - Central bank balance sheet comparison")
        logger.info("5. global_liquidity_index.png - Global liquidity index and regimes")
        logger.info("6. global_liquidity_index.csv - Raw liquidity index data")
        
        if fred_api_key is None:
            logger.info("\nNote: For enhanced analysis with official Fed data,")
            logger.info("obtain a free FRED API key from https://fred.stlouisfed.org/")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        mt5.shutdown()
