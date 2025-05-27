#!/usr/bin/env python3
"""
Script to fix syntax errors in the TradingEnv class.
"""
import os
import sys
import logging
import traceback
import shutil
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src', 'models', 'rl', 'trading_env.py')
    
    backup_path = os.path.join(os.path.dirname(env_file_path), 'trading_env_backup_syntax.py')
    shutil.copy2(env_file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    
    with open(env_file_path, 'r') as f:
        content = f.read()
    
    step_pattern = r"shares_held = float\(self\.shares_held\.iloc\[0\]\) if hasattr\(self\.shares_held, 'iloc'\) else float\(self\.shares_held\)"
    step_replacement = "shares_held = float(self.shares_held.iloc[0]) if hasattr(self.shares_held, 'iloc') and len(self.shares_held) > 0 else float(self.shares_held)"
    
    content = re.sub(step_pattern, step_replacement, content)
    
    obs_pattern = r"float\(normalized_balance\) if not isinstance\(normalized_balance, float\) else normalized_balance if not isinstance\(normalized_balance, float\) else normalized_balance"
    obs_replacement = "float(normalized_balance) if not isinstance(normalized_balance, float) else normalized_balance"
    
    content = re.sub(obs_pattern, obs_replacement, content)
    
    shares_pattern = r"float\(normalized_shares\.iloc\[0\]\) if isinstance\(normalized_shares, pd\.Series\) else float\(normalized_shares\) if not isinstance\(normalized_shares, float\) else normalized_shares if not isinstance\(normalized_shares, float\) else normalized_shares"
    shares_replacement = "float(normalized_shares.iloc[0]) if isinstance(normalized_shares, pd.Series) else float(normalized_shares) if not isinstance(normalized_shares, float) else normalized_shares"
    
    content = re.sub(shares_pattern, shares_replacement, content)
    
    portfolio_pattern = r"float\(normalized_portfolio_value\.iloc\[0\]\) if isinstance\(normalized_portfolio_value, pd\.Series\) else float\(normalized_portfolio_value\) if not isinstance\(normalized_portfolio_value, float\) else normalized_portfolio_value if not isinstance\(normalized_portfolio_value, float\) else normalized_portfolio_value"
    portfolio_replacement = "float(normalized_portfolio_value.iloc[0]) if isinstance(normalized_portfolio_value, pd.Series) else float(normalized_portfolio_value) if not isinstance(normalized_portfolio_value, float) else normalized_portfolio_value"
    
    content = re.sub(portfolio_pattern, portfolio_replacement, content)
    
    reset_pattern = r"self\.balance = float\(self\.initial_balance\)"
    reset_replacement = "self.balance = float(self.initial_balance)"
    
    content = re.sub(reset_pattern, reset_replacement, content)
    
    with open(env_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully updated {env_file_path}")
    
    with open(env_file_path, 'r') as f:
        content = f.read()
    
    content = content.replace("float(self.shares_held.iloc[0]) = 0", "self.shares_held = 0.0")
    content = content.replace("float(self.balance.iloc[0]) = ", "self.balance = ")
    
    account_info_pattern = r"""account_info = np\.array\(\[
            .*?
            .*?
            .*?
        \], dtype=np\.float32\)"""
    
    account_info_replacement = """account_info = np.array([
            float(normalized_balance) if not isinstance(normalized_balance, float) else normalized_balance,
            float(normalized_shares) if not isinstance(normalized_shares, float) else normalized_shares,
            float(normalized_portfolio_value) if not isinstance(normalized_portfolio_value, float) else normalized_portfolio_value
        ], dtype=np.float32)"""
    
    content = re.sub(account_info_pattern, account_info_replacement, content, flags=re.DOTALL)
    
    with open(env_file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"Successfully applied comprehensive fixes to {env_file_path}")
    
except Exception as e:
    logger.error(f"Error: {str(e)}")
    logger.error(traceback.format_exc())
