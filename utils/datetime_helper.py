import pandas as pd

def normalize_datetime(df, col='date_publication'):
    """
    Normalise une colonne datetime en supprimant le fuseau horaire
    """
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        if hasattr(df[col].dtype, 'tz'):
            df[col] = df[col].dt.tz_localize(None)
    return df

def safe_date_comparison(series, date_value):
    """
    Compare une série datetime avec une date de manière sécurisée
    """
    # Normaliser la série
    if hasattr(series.dtype, 'tz'):
        series = series.dt.tz_localize(None)
    
    if hasattr(date_value, 'tz'):
        date_value = date_value.tz_localize(None)
    elif not isinstance(date_value, pd.Timestamp):
        date_value = pd.Timestamp(date_value)
    
    return series >= date_value