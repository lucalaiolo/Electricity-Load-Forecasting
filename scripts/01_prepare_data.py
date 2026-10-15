"""Download the OPSD hourly time series and build the (days x 24) load table."""

from loadforecast import config, data

if __name__ == "__main__":
    path = data.download()
    table = data.build_processed()
    print(f"raw data: {path}")
    print(f"processed: {config.PROCESSED_CSV} ({len(table)} days, {table.index[0].date()} .. {table.index[-1].date()})")
