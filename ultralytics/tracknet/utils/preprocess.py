import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def preprocess_csv(csv_file):
    # Read the ball_trajectory csv file
    ball_trajectory_df = pd.read_csv(csv_file)
    ball_trajectory_df['nX'] = ball_trajectory_df['X'].shift(-1).fillna(ball_trajectory_df['X'])
    ball_trajectory_df['nY'] = ball_trajectory_df['Y'].shift(-1).fillna(ball_trajectory_df['Y'])

    if 'Event' in ball_trajectory_df.columns:
        ball_trajectory_df['hit'] = ((ball_trajectory_df['Event'] == 1) | (ball_trajectory_df['Event'] == 2)).astype(int)
    else:
        ball_trajectory_df['hit'] = 0

    #ball_trajectory_df['prev_hit'] = ball_trajectory_df['hit'].shift(fill_value=0)
    #ball_trajectory_df['next_hit'] = ball_trajectory_df['hit'].shift(-1, fill_value=0)
    #ball_trajectory_df['hit'] = ball_trajectory_df[['hit', 'prev_hit', 'next_hit']].max(axis=1)

    if 'Visibility' in ball_trajectory_df.columns:
        visibility = ball_trajectory_df['Visibility'].values
        x_vals = ball_trajectory_df['X'].values
        y_vals = ball_trajectory_df['Y'].values
        
        movement_threshold_3 = 3.0  # 判定靜止的最大首尾位移（pixel）
        movement_threshold_5 = 5.0  # 判定靜止的最大首尾位移（pixel）
        
        # 找出所有 Visibility==1 的索引
        vis_indices = np.where(visibility == 1)[0]
        if vis_indices.size > 0:
            # (a) 第一段：從第一個出現球的索引開始，向後找出連續區段
            for i in range(len(vis_indices)-1):
                dx = x_vals[vis_indices[i]] - x_vals[vis_indices[i+1]]
                dy = y_vals[vis_indices[i]] - y_vals[vis_indices[i+1]]
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= movement_threshold_5:
                    ball_trajectory_df.loc[vis_indices[i]:vis_indices[i+1], 'Visibility'] = 0
                else:
                    break

            # (b) 最後一段：從最後一個出現球的索引開始，向前找出連續區段
            # 處理最後一段：從最後一個出現的索引往回找
            for i in range(len(vis_indices)-1, 0, -1):
                dx = x_vals[vis_indices[i]] - x_vals[vis_indices[i-1]]
                dy = y_vals[vis_indices[i]] - y_vals[vis_indices[i-1]]
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= movement_threshold_5:
                    ball_trajectory_df.loc[vis_indices[i-1]:vis_indices[i], 'Visibility'] = 0
                else:
                    break

    else:
        print("Warning: 'Visibility' column not found in CSV.")

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    
    ball_trajectory_df = ball_trajectory_df.drop(drop_columns, axis=1, errors='ignore')

    return ball_trajectory_df

def compute_speed_no_smooth(df):
    speeds = [0.0]
    for i in range(1, len(df)):
        dx = df.loc[i, 'X'] - df.loc[i - 1, 'X']
        dy = df.loc[i, 'Y'] - df.loc[i - 1, 'Y']
        spd = math.sqrt(dx * dx + dy * dy)
        speeds.append(spd)
    return speeds

def filter_static_mask(df, speed_threshold=5.0, min_static_frames=5, static_radius=5):
    visible_df = df[df['Visibility'] > 0].copy()
    visible_df.reset_index(inplace=True)  # 保留原始 index
    if len(visible_df) == 0:
        return pd.Series([False] * len(df))  # 全部保留

    visible_df['speed'] = compute_speed_no_smooth(visible_df)
    visible_df['is_static'] = visible_df['speed'] < speed_threshold

    drop_indices = set()

    # 前端靜止段
    front_static_count = 0
    for is_static in visible_df['is_static']:
        if is_static:
            front_static_count += 1
        else:
            break

    if front_static_count >= min_static_frames:
        pivot = visible_df.iloc[0]
        for i in range(front_static_count):
            dist = math.hypot(visible_df.loc[i, 'X'] - pivot['X'], visible_df.loc[i, 'Y'] - pivot['Y'])
            if dist <= static_radius:
                drop_indices.add(visible_df.loc[i, 'index'])

    # 後端靜止段
    back_static_count = 0
    for is_static in reversed(visible_df['is_static'].tolist()):
        if is_static:
            back_static_count += 1
        else:
            break

    if back_static_count >= min_static_frames:
        pivot = visible_df.iloc[-1]
        for i in reversed(range(len(visible_df) - back_static_count, len(visible_df))):
            dist = math.hypot(visible_df.loc[i, 'X'] - pivot['X'], visible_df.loc[i, 'Y'] - pivot['Y'])
            if dist <= static_radius:
                drop_indices.add(visible_df.loc[i, 'index'])

    return df.index.isin(drop_indices)

def split_into_segments(df, max_missing_frames=30):
    segments = []
    start_idx = 0
    consecutive_missing = 0
    missing_run_start = -1

    for i in range(len(df)):
        vis = df.loc[i, 'Visibility']
        if vis == 0:
            if consecutive_missing == 0:
                missing_run_start = i
            consecutive_missing += 1
        else:
            if consecutive_missing >= max_missing_frames:
                segments.append(df.iloc[start_idx:missing_run_start].copy())
                start_idx = i
            consecutive_missing = 0

    if start_idx < len(df):
        segments.append(df.iloc[start_idx:].copy())

    return segments

def preprocess_csvV2(csv_path, speed_threshold=10.0, min_static_frames=5, max_missing_frames=20, static_radius=8.0):
    df_all = pd.read_csv(csv_path)

    if 'Visibility' not in df_all.columns:
        raise ValueError(f"CSV file 缺少 Visibility 欄位: {csv_path}")
    
    df_all['nX'] = df_all['X'].shift(-1).fillna(df_all['X'])
    df_all['nY'] = df_all['Y'].shift(-1).fillna(df_all['Y'])

    if 'Event' in df_all.columns:
        df_all['hit'] = ((df_all['Event'] == 1) | (df_all['Event'] == 2)).astype(int)
    else:
        df_all['hit'] = 0

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    df_all = df_all.drop(drop_columns, axis=1, errors='ignore')

    df_all['is_removed'] = False
    df_all['segment_id'] = -1  # 預設 -1，未分段時

    segments = split_into_segments(df_all, max_missing_frames=max_missing_frames)

    for seg_id, seg in enumerate(segments):
        seg_idx = seg.index
        df_all.loc[seg_idx, 'segment_id'] = seg_id

        mask = filter_static_mask(seg, speed_threshold, min_static_frames, static_radius)
        df_all.loc[seg_idx[mask], 'is_removed'] = True

        df_all['Visibility_orig'] = df_all['Visibility']
        df_all.loc[seg_idx[mask], 'Visibility'] = 0

    plot_static_removal_comparison(df_all, convert_to_static_removal_path(csv_path))

    drop_columns = ['segment_id', 'is_removed', 'Visibility_orig']
    df_all = df_all.drop(drop_columns, axis=1, errors='ignore')

    return df_all

def convert_to_static_removal_path(csv_path):
    parent_dir = os.path.dirname(csv_path)                # /path/to/dir
    static_dir = os.path.join(os.path.dirname(parent_dir), 'static_removal')
    os.makedirs(static_dir, exist_ok=True)                # 自動建立目錄（若不存在）

    base_name = os.path.splitext(os.path.basename(csv_path))[0]  # filename
    return os.path.join(static_dir, f"{base_name}.png")

def plot_static_removal_comparison(df, save_path=None):
    """
    視覺化每個 segment 的移動軌跡與被移除的靜止點：
    - 保留段連線 (線條 + 淡色點)
    - 靜止段標紅色叉叉
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Static Removal Visualization")

    segment_ids = df['segment_id'].unique()
    cmap = plt.cm.get_cmap('tab10', len(segment_ids))

    for idx, seg_id in enumerate(segment_ids):
        seg = df[df['segment_id'] == seg_id]
        seg = seg[~((seg['X'] == 0) & (seg['Y'] == 0))]
        seg_kept = seg[~seg['is_removed']]
        seg_removed = seg[seg['is_removed']]

        color = cmap(idx)

        # 1. 畫折線（軌跡）
        plt.plot(seg_kept['X'], seg_kept['Y'],
                 color=color, alpha=0.8, linewidth=1.5,
                 label=f'Segment {seg_id} Trajectory')

        # 2. 保留點（淡色圓點）
        plt.scatter(seg_kept['X'], seg_kept['Y'],
                    color=color, alpha=0.4)

        # 3. 被移除點（紅色叉叉）
        plt.scatter(seg_removed['X'], seg_removed['Y'],
                    color='red', marker='x', label=f'Segment {seg_id} Removed')

    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"圖已儲存至: {save_path}")
    else:
        plt.show()
    plt.close()
