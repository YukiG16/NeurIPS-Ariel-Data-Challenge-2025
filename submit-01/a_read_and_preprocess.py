def a_read_and_preprocess(dataset, adc_info, planet_ids):
    """Read the AIRS-CH0 files for all planet_ids and extract the time series.
    
    Parameters
    dataset: 'train' or 'test'
    adc_info: metadata dataframe, either train_adc_info or test_adc_info
    planet_ids: list of planet ids
    
    Returns
    dataframe with one row per planet_id and 5625 values per row
    
    """
    a_raw_train = np.full((len(planet_ids), 5625), np.nan, dtype=np.float32)
    for i, planet_id in tqdm(list(enumerate(planet_ids))):
        signal = pl.read_parquet(f'/kaggle/input/ariel-data-challenge-2025/{dataset}/{planet_id}/AIRS-CH0_signal_0.parquet')
        mean_signal = signal.cast(pl.Int32).sum_horizontal().cast(pl.Float32).to_numpy() / (32*356) # mean over the 32*356 pixels
        net_signal = mean_signal[1::2] - mean_signal[0::2]
        a_raw_train[i] = net_signal
    return a_raw_train
    
