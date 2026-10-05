import numpy as np

def calculate_rmse(obs, forecast):
    return np.sqrt(np.mean((obs - forecast) ** 2))

def calculate_mae(obs, forecast):
    return np.mean(np.abs(obs - forecast))

def calculate_contingency(obs, forecast, threshold=64.5):
    obs_event = obs >= threshold
    fcst_event = forecast >= threshold
    
    hits = np.sum(obs_event & fcst_event)
    misses = np.sum(obs_event & ~fcst_event)
    false_alarms = np.sum(~obs_event & fcst_event)
    correct_negatives = np.sum(~obs_event & ~fcst_event)
    
    return hits, misses, false_alarms, correct_negatives

def calculate_csi(hits, misses, false_alarms):
    denominator = hits + misses + false_alarms
    return hits / denominator if denominator > 0 else 0.0

def calculate_pod(hits, misses):
    denominator = hits + misses
    return hits / denominator if denominator > 0 else 0.0

def calculate_far(hits, false_alarms):
    denominator = hits + false_alarms
    return false_alarms / denominator if denominator > 0 else 0.0

def calculate_ets(hits, misses, false_alarms, correct_negatives):
    total = hits + misses + false_alarms + correct_negatives
    if total == 0:
        return 0.0
    
    hits_random = ((hits + misses) * (hits + false_alarms)) / total
    denominator = hits + misses + false_alarms - hits_random
    return (hits - hits_random) / denominator if denominator > 0 else 0.0

def calculate_fss_proxy(obs, forecast, threshold=64.5):
    """
    Simplified proxy for Fractions Skill Score without actual spatial grids.
    Calculates moving window fraction matching (1D array simulation).
    """
    obs_event = (obs >= threshold).astype(float)
    fcst_event = (forecast >= threshold).astype(float)
    
    window = 5
    fss_values = []
    
    for i in range(len(obs) - window):
        o_frac = np.mean(obs_event[i:i+window])
        f_frac = np.mean(fcst_event[i:i+window])
        
        mse = (o_frac - f_frac) ** 2
        mse_ref = o_frac ** 2 + f_frac ** 2
        fss_values.append(1 - (mse / mse_ref) if mse_ref > 0 else 1.0)
        
    return np.mean(fss_values) if fss_values else 0.0
