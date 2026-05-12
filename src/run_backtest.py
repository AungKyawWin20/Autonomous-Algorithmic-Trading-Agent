from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest_strategy import (
    load_model_and_artifacts,
    load_historical_data,
    run_backtest,
    calculate_metrics,
    plot_results,
    INITIAL_CAPITAL
)

def main():
    # backtest period
    START_DATE = datetime(2018, 1, 1)
    END_DATE = datetime(2018, 12, 31)
    
    print("=" * 60)
    print("LSTM Trading Strategy Backtest")
    print(f"Period: {START_DATE.date()} to {END_DATE.date()}")
    print("=" * 60)
  
    model, scaler, feature_cols = load_model_and_artifacts()
    # run backtest
    trades_df, portfolio_df = run_backtest(
        START_DATE, END_DATE, model, scaler, feature_cols
    )
    # calculate metrics
    metrics = calculate_metrics(trades_df, portfolio_df)
    
    # results printout
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Initial Capital: ${metrics.get('initial_capital', 0):,.2f}")
    print(f"Final Value: ${metrics.get('final_value', 0):,.2f}")
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"\nSharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"\nTotal Trades: {metrics.get('total_trades', 0)}")
    print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
    print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
    print(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"Average Return: {metrics.get('avg_return_pct', 0):.2f}%")
    print(f"Average Win: {metrics.get('avg_win_pct', 0):.2f}%")
    print(f"Average Loss: {metrics.get('avg_loss_pct', 0):.2f}%")
    

    output_dir = PROJECT_ROOT / "backtest_results"
    output_dir.mkdir(exist_ok=True)
    
    period_label = f"{START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}"
    trades_df.to_csv(output_dir / f"Trades ({period_label}).csv", index=False)
    portfolio_df.to_csv(output_dir / f"Portfolio ({period_label}).csv", index=False)
    
    print(f"\n Results saved to {output_dir}")
    
    # result plots
    print("plots...")
    fig = plot_results(portfolio_df, trades_df, metrics)
    fig.savefig(output_dir / f"BacktestResults ({period_label}).png", dpi=150, bbox_inches='tight')
    print(f"Plot saved")
    
    return trades_df, portfolio_df, metrics

if __name__ == "__main__":
    main()

