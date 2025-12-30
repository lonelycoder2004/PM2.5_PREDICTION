"""
verify_collections.py

Simple helper to inspect both 'predictions' and 'newpredictions' collections and print statistics.
"""
import database
import pandas as pd


def show_collections():
    stats = database.count_records()
    print("Collection counts:")
    print(f"  predictions: {stats.get('predictions', 0)}")
    print(f"  newpredictions: {stats.get('newpredictions', 0)}\n")

    print("newpredictions stats:")
    lag_stats = database.get_lag_collection_stats()
    if lag_stats.get('total', 0) == 0:
        print("  No historical data in 'newpredictions'.")
    else:
        print(f"  total records: {lag_stats['total']}")
        print(f"  unique locations: {lag_stats['unique_locations']}")
        print(f"  PM2.5 min: {lag_stats['pm25_min']:.2f}")
        print(f"  PM2.5 max: {lag_stats['pm25_max']:.2f}")
        print(f"  PM2.5 avg: {lag_stats['pm25_avg']:.2f}")

    # show few sample docs from newpredictions
    col = database.get_lag_collection()
    sample = list(col.find().limit(5))
    if sample:
        print('\nSample newpredictions records:')
        df = pd.DataFrame(sample)
        print(df[['observation_time','latitude','longitude','pm25']].head())

    # show few sample docs from predictions
    colp = database.get_collection()
    samplep = list(colp.find().limit(5))
    if samplep:
        print('\nSample predictions records:')
        dfp = pd.DataFrame(samplep)
        # If timestamp is present, show it
        cols = ['timestamp','observation_time','latitude','longitude','pm25','source']
        cols_present = [c for c in cols if c in dfp.columns]
        print(dfp[cols_present].head())


if __name__ == '__main__':
    database.init_db()
    show_collections()
