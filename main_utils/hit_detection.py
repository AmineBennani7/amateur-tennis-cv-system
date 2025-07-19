import pandas as pd
from scipy.signal import find_peaks

def group_close_peaks(peaks, signal, max_frame_gap=20):
    if not peaks:
        return []
    grouped = []
    current_group = [peaks[0]]
    for p in peaks[1:]:
        if p - current_group[-1] <= max_frame_gap:
            current_group.append(p)
        else:
            best_peak = max(current_group, key=lambda x: abs(signal[x]))
            grouped.append(best_peak)
            current_group = [p]
    best_peak = max(current_group, key=lambda x: abs(signal[x]))
    grouped.append(best_peak)
    return grouped

def filter_small_vertical_jumps(peaks, signal, min_vertical_jump=100):
    if not peaks:
        return []
    filtered = [peaks[0]]
    for i in range(1, len(peaks)):
        prev_peak = filtered[-1]
        curr_peak = peaks[i]
        vertical_jump = abs(signal[curr_peak] - signal[prev_peak])
        if vertical_jump >= min_vertical_jump:
            filtered.append(curr_peak)
    return filtered

def detect_hits(ball_positions_all):
    y_positions = [pos[1] if pos[1] > 0 else -1 for pos in ball_positions_all]
    df_ball_positions = pd.DataFrame({'y1': y_positions})
    df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['y1'].rolling(window=5, min_periods=1).mean()
    signal = df_ball_positions['mid_y_rolling_mean'].values

    MIN_DISTANCE = 25
    MIN_PROMINENCE = 130

    peaks_max, _ = find_peaks(signal, distance=MIN_DISTANCE, prominence=MIN_PROMINENCE)
    peaks_min, _ = find_peaks(-signal, distance=MIN_DISTANCE, prominence=MIN_PROMINENCE)
    all_peaks = sorted(list(peaks_max) + list(peaks_min))

    final_hits = group_close_peaks(all_peaks, signal, max_frame_gap=20)
    final_hits_filtered = filter_small_vertical_jumps(final_hits, signal, min_vertical_jump=120)

    return final_hits_filtered
