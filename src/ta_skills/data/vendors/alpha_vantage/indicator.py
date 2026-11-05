import logging

from ta_skills.data.errors import NoMarketDataError, VendorError
from ta_skills.data.indicator_registry import effective_params, get_spec
from ta_skills.data.vendors.alpha_vantage.common import _make_api_request

logger = logging.getLogger(__name__)


def get_indicator(
    symbol: str,
    indicator: str,
    as_of_date: str,
    look_back_days: int,
    interval: str = "daily",
    series_type: str = "close"
) -> str:
    """
    Returns Alpha Vantage technical indicator values over a time window.

    The endpoint, column and parameters come from the indicator registry, and a
    configured ``indicator_params`` window is sent as the time period, so this
    vendor and Yahoo compute the same indicator.

    Args:
        symbol: ticker symbol of the company
        indicator: technical indicator to get the analysis and report of
        as_of_date: The current trading date you are trading on, YYYY-mm-dd
        look_back_days: how many days to look back
        interval: Time interval (daily, weekly, monthly)
        series_type: The desired price type (close, open, high, low)

    Returns:
        String containing indicator values and description
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    spec = get_spec(indicator)  # unknown names raise ValueError

    # Names Alpha Vantage cannot serve (vwma, the custom OHLCV indicators, ...)
    # must RAISE: route_to_vendor only advances its fallback chain on an
    # exception, so returning prose here would short-circuit the yfinance
    # fallback and the agent would never get the indicator values. A vendor
    # error, not a caller error: another vendor may compute it.
    if spec.av_function is None:
        raise NoMarketDataError(
            symbol, symbol, f"Alpha Vantage does not serve the {indicator} indicator"
        )

    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")
    before = as_of_dt - relativedelta(days=look_back_days)

    params = {
        "symbol": symbol,
        "interval": interval,
        "datatype": "csv",
    }
    if spec.av_series_type:
        params["series_type"] = series_type
    if spec.av_takes_window:
        params["time_period"] = str(effective_params(indicator)["window"])
    params.update(spec.av_fixed_params)

    try:
        data = _make_api_request(spec.av_function, params)

        lines = data.strip().split('\n')
        if len(lines) < 2:
            return f"Error: No data returned for {indicator}"

        header = [col.strip() for col in lines[0].split(',')]
        try:
            date_col_idx = header.index('time')
        except ValueError:
            return f"Error: 'time' column not found in data for {indicator}. Available columns: {header}"

        if not spec.av_column:
            # Default to the second column if no specific mapping exists
            value_col_idx = 1
        else:
            try:
                value_col_idx = header.index(spec.av_column)
            except ValueError:
                return f"Error: Column '{spec.av_column}' not found for indicator '{indicator}'. Available columns: {header}"

        result_data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            values = line.split(',')
            if len(values) > value_col_idx:
                try:
                    date_str = values[date_col_idx].strip()
                    date_dt = datetime.strptime(date_str, "%Y-%m-%d")

                    if before <= date_dt <= as_of_dt:
                        value = values[value_col_idx].strip()
                        result_data.append((date_dt, value))
                except (ValueError, IndexError):
                    continue

        # Sort by date and format output
        result_data.sort(key=lambda x: x[0])

        ind_string = ""
        for date_dt, value in result_data:
            ind_string += f"{date_dt.strftime('%Y-%m-%d')}: {value}\n"

        if not ind_string:
            ind_string = "No data available for the specified date range.\n"

        result_str = (
            f"## {indicator.upper()} values from {before.strftime('%Y-%m-%d')} to {as_of_date}:\n\n"
            + ind_string
            + "\n\n"
            + spec.description
        )

        return result_str

    except VendorError:
        # Unavailable vendor, throttle, or an indicator this vendor does not
        # serve. Let it propagate so the router falls back to a vendor that can,
        # instead of returning the failure as a successful-looking string.
        raise
    except Exception as e:
        logger.warning("Alpha Vantage indicator %s failed: %s", indicator, e)
        raise NoMarketDataError(symbol, symbol, f"{indicator} unavailable: {e}") from e
