import os
import sys
import re
from datetime import datetime
from pathlib import Path
from io import StringIO

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = ROOT / "backtest_results"


def run_backtest_cli(start_date: str, end_date: str) -> tuple[str, str, int]:
    """Run backtest with custom date range and capture output."""
    from src.backtest_strategy import (
        load_model_and_artifacts,
        load_historical_data,
        run_backtest,
        calculate_metrics,
        plot_results,
        INITIAL_CAPITAL
    )
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Run backtest
        model, scaler, feature_cols = load_model_and_artifacts()
        trades_df, portfolio_df = run_backtest(start, end, model, scaler, feature_cols)
        metrics = calculate_metrics(trades_df, portfolio_df)
        
        # Save results
        period_label = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        BACKTEST_DIR.mkdir(exist_ok=True)
        trades_df.to_csv(BACKTEST_DIR / f"Trades ({period_label}).csv", index=False)
        portfolio_df.to_csv(BACKTEST_DIR / f"Portfolio ({period_label}).csv", index=False)
        
        # Generate and save plot
        fig = plot_results(portfolio_df, trades_df, metrics)
        fig.savefig(BACKTEST_DIR / f"BacktestResults ({period_label}).png", dpi=150, bbox_inches='tight')
        
        returncode = 0
        stderr = ""
        captured_output.write(f"\nBacktest completed successfully!")
        captured_output.write(f"\nResults saved to {BACKTEST_DIR}")
    except Exception as e:
        captured_output.write(f"Error: {str(e)}")
        returncode = 1
        stderr = str(e)
    finally:
        sys.stdout = old_stdout
    
    stdout = captured_output.getvalue()
    return stdout, stderr, returncode


def run_alpaca_trader(dry_run: bool) -> tuple[str, str, int]:
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()

    try:
        # Import and run the trader
        from src.alpaca_trader import run as run_trader

        serpapi_key = os.environ.get("SERP_API")
        tiingo_token = os.environ.get("Tiingo_API")

        if not serpapi_key:
            raise ValueError("SERP_API environment variable not set.")
        if not tiingo_token:
            raise ValueError("Tiingo_API environment variable not set.")

        run_trader(
            serpapi_key=serpapi_key,
            tiingo_token=tiingo_token,
            dry_run=dry_run
        )
        returncode = 0
        stderr = ""
    except Exception as e:
        captured_output.write(f"Error: {str(e)}")
        returncode = 1
        stderr = str(e)
    finally:
        sys.stdout = old_stdout

    stdout = captured_output.getvalue()
    return stdout, stderr, returncode


def display_trader_output(stdout: str) -> None:
    lines = stdout.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if 'Updating price data' in line:
            st.subheader("Step 1: Updating Price Data")
            st.info(line)
        elif 'DRY RUN' in line:
            st.warning(line)
        elif 'Would have bought' in line:
            st.info(line)
        elif 'Closing existing positions' in line:
            st.subheader("Step 2: Closing Positions")
            st.info(line)
        elif 'Waiting' in line and 'settle' in line:
            st.info(line)
        elif 'Placing new orders' in line:
            st.subheader("Step 3: Placing Orders")
            st.info(line)
        elif 'Portfolio value' in line:
            match = re.search(r'\$([\d,]+\.\d+)', line)
            if match:
                value = match.group(1)
                st.metric("Portfolio Value", f"${value}")
        elif 'Cash available' in line:
            match = re.search(r'\$([\d,]+\.\d+)', line)
            if match:
                value = match.group(1)
                st.metric("Cash Available", f"${value}")
        elif 'Per position' in line:
            match = re.search(r'\$([\d,]+\.\d+)', line)
            if match:
                value = match.group(1)
                st.metric("Amount per Position", f"${value}")
        elif 'Buying' in line:
            st.info(line)
        elif '[' in line and 'BUY' in line:
            st.success(line)
        elif 'Order ID' in line:
            st.success(line)
        elif 'WEEKLY RUN COMPLETE' in line:
            st.subheader("Run Complete")
            st.success(line)
        elif 'Signals generated' in line:
            st.info(line)
        elif 'Orders placed' in line:
            st.info(line)
        elif 'Skipped' in line:
            st.warning(line)
        elif 'No open positions' in line:
            st.info(line)
        elif 'No BUY signals' in line:
            st.info(line)
        else:
            st.write(line)


def list_backtest_runs() -> list[str]:
    png_files = sorted(BACKTEST_DIR.glob("BacktestResults *.png"))
    runs = [p.stem.replace("BacktestResults (", "").replace(")", "") for p in png_files]
    return runs


def load_backtest_csv(run_id: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    portfolio_path = BACKTEST_DIR / f"Portfolio ({run_id}).csv"
    trades_path = BACKTEST_DIR / f"Trades ({run_id}).csv"
    portfolio_df = None
    trades_df = None

    try:
        if portfolio_path.exists():
            portfolio_df = pd.read_csv(portfolio_path, parse_dates=["date"])
    except Exception as e:
        st.error(f"Error loading portfolio CSV: {e}")
        portfolio_df = None
    
    try:
        if trades_path.exists():
            trades_df = pd.read_csv(trades_path, parse_dates=["date"])
    except Exception as e:
        st.error(f"Error loading trades CSV: {e}")
        trades_df = None

    return portfolio_df, trades_df


def render_backtest_summary(portfolio_df: pd.DataFrame, trades_df: pd.DataFrame) -> None:
    # Validate portfolio dataframe
    if portfolio_df is None or portfolio_df.empty:
        st.error("Portfolio dataframe is empty.")
        return
    
    if "portfolio_value" not in portfolio_df.columns:
        st.error("Portfolio dataframe missing 'portfolio_value' column.")
        return
    
    initial_value = float(portfolio_df["portfolio_value"].iloc[0])
    final_value = float(portfolio_df["portfolio_value"].iloc[-1])
    total_return = (final_value - initial_value) / initial_value * 100
    drawdown = portfolio_df["drawdown"].min() if "drawdown" in portfolio_df.columns else None

    # Handle empty trades dataframe
    if trades_df is None or trades_df.empty:
        trade_returns = pd.Series(dtype=float)
    else:
        trade_returns = trades_df["pnl_pct"].dropna() if "pnl_pct" in trades_df.columns else pd.Series(dtype=float)
    
    win_count = int((trade_returns > 0).sum())
    total_trades = int(len(trade_returns))
    win_rate = (win_count / total_trades * 100) if total_trades else 0.0
    avg_return = float(trade_returns.mean()) if not trade_returns.empty else 0.0

    st.markdown("### Backtest Summary")
    st.metric("Initial capital", f"${initial_value:,.2f}")
    st.metric("Final capital", f"${final_value:,.2f}")
    st.metric("Total return", f"{total_return:.2f}%")
    if drawdown is not None:
        st.metric("Max drawdown", f"{drawdown:.2f}%")

    st.markdown(
        f"- Trades analyzed: **{total_trades}**\n"
        f"- Win rate: **{win_rate:.1f}%**\n"
        f"- Avg return per trade: **{avg_return:.2f}%**"
    )


def main() -> None:
    st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

    # authentication
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("Login to AI Trading Dashboard")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "iamdeadinside":
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")
        return

    # logged in
    st.sidebar.button("Logout", on_click=lambda: st.session_state.update(logged_in=False))

    st.title("AI Trading Dashboard")

    # Add some custom CSS for better styling
    st.markdown("""
    <style>
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
    }
    .stSubheader {
        color: #2c3e50;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Select page", ["Alpaca Trader", "Run Backtest", "Backtest Results"])

    if page == "Alpaca Trader":
        st.header("Run Alpaca Trading Workflow")
        st.write(
            "Execute the automated trading workflow using AI signals. "
            "This will update data, generate signals, and place orders on Alpaca."
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            dry_run = st.checkbox("Dry run (do not place orders)", value=True)
        with col2:
            run_button = st.button("Run Alpaca Trader", use_container_width=True)

        st.markdown("---")

        if run_button:
            with st.spinner("Executing Alpaca workflow..."):
                stdout, stderr, returncode = run_alpaca_trader(dry_run=dry_run)

            if returncode == 0:
                st.success("Alpaca trader finished successfully.")
            else:
                st.error(f"Trader finished with exit code {returncode}.")
                if stderr:
                    st.error(f"Error details: {stderr}")

            if stdout:
                display_trader_output(stdout)

        st.sidebar.info(
            "Ensure environment variables are set: SERP_API, Tiingo_API, Alpaca_Key, Alpaca_Secret."
        )

    elif page == "Run Backtest":
        st.header("Run Backtest")
        st.write(
            "Run a backtest of the LSTM trading strategy for a custom date range. "
            "Results will be saved to the backtest_results folder."
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime(2022, 1, 1))
        with col2:
            end_date = st.date_input("End Date", value=datetime(2022, 12, 31))

        run_backtest_btn = st.button("Run Backtest", use_container_width=True)

        if run_backtest_btn:
            if start_date >= end_date:
                st.error("Start date must be before end date.")
            else:
                with st.spinner("Running backtest... this may take a few minutes."):
                    stdout, stderr, returncode = run_backtest_cli(
                        start_date=str(start_date),
                        end_date=str(end_date)
                    )

                if returncode == 0:
                    st.success("Backtest completed successfully!")
                else:
                    st.error(f"Backtest finished with exit code {returncode}.")
                    if stderr:
                        st.error(f"Error details: {stderr}")

                if stdout:
                    st.text(stdout)

                # Show the results
                period_label = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                result_image = BACKTEST_DIR / f"BacktestResults ({period_label}).png"

                if result_image.exists():
                    st.image(str(result_image), caption=f"Backtest results", use_column_width=True)
                    
                    portfolio_df, trades_df = load_backtest_csv(period_label)
                    if portfolio_df is not None and trades_df is not None:
                        render_backtest_summary(portfolio_df, trades_df)

    else:
        st.header("Backtest Results")
        st.write("Browse the generated backtest summary charts and examine the portfolio and trade CSV data.")

        runs = list_backtest_runs()
        if not runs:
            st.warning("No backtest result files were found in the `backtest_results` folder.")
            return

        run_id = st.selectbox("Select a backtest run", runs)
        if run_id:
            result_image = BACKTEST_DIR / f"BacktestResults ({run_id}).png"
            portfolio_df, trades_df = load_backtest_csv(run_id)

            if result_image.exists():
                st.image(str(result_image), caption=f"Backtest summary for {run_id}", use_column_width=True)
            else:
                st.warning(f"Summary image not found for run {run_id}.")

            if portfolio_df is not None and trades_df is not None:
                render_backtest_summary(portfolio_df, trades_df)
                st.markdown("---")

                tab1, tab2 = st.tabs(["Portfolio data", "Trades data"])
                with tab1:
                    st.subheader("Portfolio data")
                    st.dataframe(portfolio_df.head(20), use_container_width=True)
                    st.download_button(
                        label="Download portfolio CSV",
                        data=portfolio_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"Portfolio ({run_id}).csv",
                        mime="text/csv",
                    )

                with tab2:
                    st.subheader("Trades data")
                    st.dataframe(trades_df.head(20), use_container_width=True)
                    st.download_button(
                        label="Download trades CSV",
                        data=trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"Trades ({run_id}).csv",
                        mime="text/csv",
                    )
            else:
                st.error("Could not load portfolio or trades data for this run.")


if __name__ == "__main__":
    main()
