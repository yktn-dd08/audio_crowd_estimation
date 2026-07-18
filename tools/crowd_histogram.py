import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from alembic.operations.toimpl import drop_index
from tqdm import tqdm
from datetime import timedelta
from shapely.geometry import LineString, Point
from shapely.wkt import loads as wkt_loads
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common.logger import get_logger


logger = get_logger('tools.crowd_histogram')
DISTANCE_CAND = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]

def decompose_linestring(
        data_row: pd.Series,
        dt: float
):
    """
    rowのLineStringを個々の点に分解し、タイムスタンプを付与する。

    Parameters
    ----------
    data_row : pd.Series
        A row from the DataFrame containing 'id', 'start_time' and 'geom' (LineString).
    dt : float
        Time interval between points.

    Returns
    -------
    list of dict
        A list of dictionaries, each containing 'time', 'x', and 'y'.
    """
    start_time = data_row['start_time']
    geom = data_row['geom']
    pid = data_row['id']

    if not isinstance(geom, LineString):
        return []

    points = []
    for i, point in enumerate(geom.coords):
        time = start_time + pd.Timedelta(seconds=i * dt)
        points.append({'id': pid, 'time': time, 'geom': Point(point[0], point[1])})

    return pd.DataFrame(points)


def read_trajectory_csv(
        csv_path: str,
        dt: float = 1.0
):
    """
    csv_pathから軌跡データを読み込み、LineStringを個々の点に分解する。
    Parameters
    ----------
    csv_path: str
        Path to the CSV file containing trajectory data with 'id', 'start_time', and 'geom' (LineString).
    dt: float
        Time interval between points.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing 'id', 'time', and 'geom' (Point) for each decomposed point.
    """
    df = pd.read_csv(csv_path)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['geom'] = df['geom'].apply(lambda x: wkt_loads(x) if isinstance(x, str) else None)
    pnt_df = pd.concat([decompose_linestring(row, dt) for _, row in df.iterrows()], ignore_index=True)
    return pnt_df


def histogram_with_distance(
        trj_df: pd.DataFrame,
        center_point: Point,
        dist_range: tuple = (0, 30),
        distance_bin: float = 0.5,
        time_bin: int = 1,
        log_scale: bool = True
):
    """
    各時刻の点の距離を計算し、距離ごとのヒストグラムを作成する。
    Parameters
    ----------
    trj_df: pd.DataFrame
        A DataFrame containing 'time' and 'geom' (Point) for each point in the trajectory.
    center_point: Point
        The reference point from which distances are calculated.
    dist_range: tuple[float]
        The range of distances to consider for the histogram (min, max).
    distance_bin: float
        The width of each distance bin for the histogram.
    time_bin: int
        The time interval (in seconds) over which to aggregate the histogram counts.
    log_scale: bool
        Whether to apply log1p transformation to the histogram counts.

    Returns
    -------
    pd.DataFrame, np.ndarray
        A DataFrame containing 'time' and 'histogram' (counts per distance bin) for each time interval,
        and an array of distance bin edges.
    """
    g_df = pd.DataFrame(
        [
            {
                'time': t,
                'geom': grp_df['geom'].tolist(),
                'distance': [center_point.distance(p) for p in grp_df['geom'].tolist()],
            }
            for t, grp_df in trj_df.groupby('time')
        ]
    )
    x_arr = np.arange(dist_range[0], dist_range[1] + distance_bin, distance_bin)
    g_df['count'] = g_df['distance'].apply(
        lambda x: np.histogram(x, bins=x_arr)[0]
    )
    crowd_max = g_df['count'].apply(lambda x: max(x) if len(x) > 0 else 0).max()
    time_list = g_df['time'].tolist()
    hist_df = pd.DataFrame(
        [
            {
                'time': tl,
                'histogram': g_df[(g_df['time'] >= tl) &
                                  (g_df['time'] < tl + timedelta(seconds=time_bin))]['count'].sum() / time_bin
            }
            for tl in time_list
        ]
    )
    # if log_scale:
    #     hist_df['histogram'] = hist_df['histogram'].apply(lambda x: np.log1p(x))
    return hist_df, x_arr


def histogram_split_by_threshold(
        trj_df: pd.DataFrame,
        center_point: Point,
        threshold: float = 3.0,
        time_bin: int = 1,
):
    """
    各時刻の点の距離を計算し、距離ごとの密度分布を作成する。
    Parameters
    ----------
    trj_df: pd.DataFrame
        A DataFrame containing 'time' and 'geom' (Point) for each point in the trajectory.
    center_point: Point
        The reference point from which distances are calculated.
    threshold: float
        The distance threshold to split the histogram into two parts (target and noise).
    time_bin: int
        The time interval (in seconds) over which to aggregate the histogram counts.

    Returns
    -------
    pd.DataFrame, np.ndarray
        A DataFrame containing 'time' and 'histogram' (counts per distance bin) for each time interval,
        and an array of distance bin edges.
    """
    g_df = pd.DataFrame(
        [
            {
                'time': t,
                'geom': grp_df['geom'].tolist(),
                'distance': [center_point.distance(p) for p in grp_df['geom'].tolist()],
            }
            for t, grp_df in trj_df.groupby('time')
        ]
    )
    g_df['target_count'] = g_df['distance'].apply(
        lambda x: len([d for d in x if d <= threshold])
    )
    g_df['noise_count'] = g_df['distance'].apply(
        lambda x: len([d for d in x if d > threshold])
    )
    time_list = g_df['time'].tolist()
    hist_df = pd.DataFrame(
        [
            {
                'time': tl,
                'noise_count': g_df[(g_df['time'] >= tl) &
                                    (g_df['time'] < tl + timedelta(seconds=time_bin))]['noise_count'].sum() / time_bin,
                'target_count': g_df[(g_df['time'] >= tl) &
                                     (g_df['time'] < tl + timedelta(seconds=time_bin))]['target_count'].sum() / time_bin
            }
            for tl in time_list
        ]
    )
    return hist_df

def calculate_histogram_index(
        distance_list: list,
        threshold: float = 3.0
):
    target_list = [1.0 / d**2 for d in distance_list if 0 < d <= threshold]
    noise_list = [1.0 / d**2 for d in distance_list if d > threshold]
    target_index = sum(target_list) / len(target_list) if len(target_list) > 0 else 0.0
    noise_index = sum(noise_list) / len(noise_list) if len(noise_list) > 0 else 0.0
    return target_index, noise_index


def index_histogram(
        trj_df: pd.DataFrame,
        center_point: Point,
        time_bin: int = 1,
        threshold: float = 3.0
):
    g_df = pd.DataFrame(
        [
            {
                'time': t,
                'geom': grp_df['geom'].tolist(),
                'distance': [center_point.distance(p) for p in grp_df['geom'].tolist()],
            }
            for t, grp_df in trj_df.groupby('time')
        ]
    )
    g_df['target_index'], g_df['noise_index'] = zip(*g_df['distance'].apply(
        lambda x: calculate_histogram_index(x, threshold=threshold)
    ))
    time_list = g_df['time'].tolist()
    hist_df = pd.DataFrame(
        [
            {
                'time': tl,
                'noise_index': g_df[(g_df['time'] >= tl) &
                                     (g_df['time'] < tl + timedelta(seconds=time_bin))]['noise_index'].sum() / time_bin,
                'target_index': g_df[(g_df['time'] >= tl) &
                                     (g_df['time'] < tl + timedelta(seconds=time_bin))]['target_index'].sum() / time_bin
            }
            for tl in time_list
        ]
    )
    return hist_df


def output_movie(
        hist_df: pd.DataFrame,
        x_arr: np.ndarray,
        output_mp4: str,
        log_scale: bool = True,
        fps: int = 10
):
    """
    ヒストグラムデータをアニメーションとして出力する。
    Parameters
    ----------
    hist_df: pd.DataFrame
        A DataFrame containing 'time' and 'histogram' (counts per distance bin) for each time interval.
    x_arr: np.ndarray
        An array of distance bin edges.
    output_mp4: str
        Path to the output MP4 file.
    log_scale: bool
        Whether the histogram counts are in log scale.
    fps: int
        Frames per second for the output video.

    Returns
    -------
    None
    """
    bin_left = x_arr[:-1]
    bin_width = np.diff(x_arr)

    max_count = hist_df['histogram'].apply(
        lambda h: np.max(h) if len(h) > 0 else 0
    ).max()
    # if log_scale:
    #     max_count = np.expm1(max_count)
    # 目盛り候補
    candidates = np.array(DISTANCE_CAND)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        bin_left,
        hist_df.iloc[0]['histogram'],
        width=bin_width,
        align='edge',
        edgecolor='black'
    )

    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_ylim(0, max_count * 1.1 if max_count > 0.0 else 1.0)
    # if not log_scale:
    #     ax.set_ylim(0, max_count * 1.1 if max_count > 0 else 1)
    # else:
    #     ax.set_ylim(0, np.log1p(max_count) * 1.1 if np.log1p(max_count) > 0 else 1)
    ax.set_xlabel("Distance from center point")
    ax.set_ylabel("Crowd count" + "Crowd count (log1p scale)" if log_scale else "")
    ax.set_title("Crowd distance histogram")
    major_ticks = np.arange(0, x_arr[-1] + 0.1, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(x)) for x in major_ticks])
    # if log_scale:
    #     y_ticks = candidates[candidates <= max_count]
    #
    #     ax.set_yticks(np.log1p(y_ticks))
    #     ax.set_yticklabels([str(v) for v in y_ticks])

    ax.grid(True, axis="y", alpha=0.3)

    time_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top'
    )

    def update(frame_idx):
        row = hist_df.iloc[frame_idx]
        hist = row['histogram']

        for bar, height in zip(bars, hist):
            bar.set_height(height)

        time_text.set_text(f"time: {row['time']}")
        return list(bars) + [time_text]

    anime = FuncAnimation(
        fig,
        update,
        frames=len(hist_df),
        interval=1000,
        blit=True
    )

    dir_name = os.path.dirname(output_mp4)
    os.makedirs(dir_name, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    with tqdm(
        total=len(hist_df),
        desc='Rendering animation',
        unit='frame'
    ) as pbar:
        anime.save(
            output_mp4,
            writer=writer,
        progress_callback=lambda current_frame, total_frames: pbar.update(current_frame + 1 - pbar.n)
        )

    plt.close(fig)
    return


def output_boxplot(
        hist_df: pd.DataFrame,
        x_arr: np.ndarray,
        output_png: str,
        log_scale: bool = True,
        remove_zero: bool = True,
):
    """
    ヒストグラムデータを箱ひげ図として出力する。
    Parameters
    ----------
    hist_df: pd.DataFrame
        A DataFrame containing 'time' and 'histogram' (counts per distance bin) for each time interval.
    x_arr: np.ndarray
        An array of distance bin edges.
    output_png: str
        Path to the output PNG file.
    log_scale: bool
        Whether to apply log1p transformation to the histogram counts.
    remove_zero: bool
        Whether to remove zero counts from the histogram before plotting.

    Returns
    -------
    None
    """
    if remove_zero:
        hist_df_ = hist_df[hist_df['histogram'].apply(lambda x: np.any(x > 0))].reset_index(drop=True)
    else:
        hist_df_ = hist_df
    if log_scale:
        hist_df_['histogram'] = hist_df_['histogram'].apply(lambda x: np.log1p(x))
    hist_matrix = np.vstack(hist_df_['histogram'].values)

    bin_centers = (x_arr[:-1] + x_arr[1:]) / 2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        hist_matrix,
        positions=bin_centers,
        widths=np.diff(x_arr) * 0.7,
        meanline=True,
        showfliers=True,
    )

    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_xlabel("Distance from center point")
    ax.set_ylabel("Crowd count" + (" (log1p scale)" if log_scale else ""))
    ax.set_title("Crowd count distribution by distance bin")
    major_ticks = np.arange(0, x_arr[-1] + 0.1, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(x)) for x in major_ticks])
    if log_scale:
        max_count = hist_matrix.max()
        max_count = np.expm1(max_count)

        # 目盛り候補
        candidates = np.array(DISTANCE_CAND)

        y_ticks = candidates[candidates <= max_count]

        ax.set_yticks(np.log1p(y_ticks))
        ax.set_yticklabels([str(v) for v in y_ticks])

    ax.grid(True, axis="y", alpha=0.3)


    fig.tight_layout()
    dir_name = os.path.dirname(output_png)
    os.makedirs(dir_name, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return


def output_mean_histogram(
        hist_df: pd.DataFrame,
        x_arr: np.ndarray,
        output_png: str,
        log_scale: bool = True,
        remove_zero: bool = True,
):
    """
    ヒストグラムの平均値の棒グラフを出力する。
    Parameters
    ----------
    hist_df: pd.DataFrame
        A DataFrame containing 'time' and 'histogram' (counts per distance bin) for each time interval.
    x_arr: np.ndarray
        An array of distance bin edges.
    output_png: str
        Path to the output PNG file.
    log_scale: bool
        Whether to apply log1p transformation to the histogram counts.
    remove_zero: bool
        Whether to remove zero counts from the histogram before plotting.
    Returns
    -------
    None
    """
    if remove_zero:
        hist_df_ = hist_df[hist_df['histogram'].apply(lambda x: np.any(x > 0))].reset_index(drop=True)
    else:
        hist_df_ = hist_df
    if log_scale:
        hist_df_['histogram'] = hist_df_['histogram'].apply(lambda x: np.log1p(x))
    mean_hist = np.mean(np.vstack(hist_df_['histogram'].values), axis=0)

    # 棒グラフを作成
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x_arr[:-1],
        mean_hist,
        width=np.diff(x_arr),
        align='edge',
        edgecolor='black'
    )
    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_xlabel("Distance from center point")
    ax.set_ylabel("Crowd count" + (" (log1p scale)" if log_scale else ""))
    ax.set_title("Crowd count distribution by distance bin")
    major_ticks = np.arange(0, x_arr[-1] + 0.1, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(x)) for x in major_ticks])
    if log_scale:
        max_count = mean_hist.max()
        max_count = np.expm1(max_count)

        # 目盛り候補
        candidates = np.array(DISTANCE_CAND)

        y_ticks = candidates[candidates <= max_count]
        # 値が小さくて目盛りが1つしかない場合は、最大値を基準に目盛りを作る
        if len(y_ticks) < 2:
            max_range = 10 ** int(np.log10(max_count))
            y_ticks = [j / 5.0 * max_range for j in range(6)]

        ax.set_yticks(np.log1p(y_ticks))
        ax.set_yticklabels([str(v) for v in y_ticks])

    ax.grid(True, axis="y", alpha=0.3)


    fig.tight_layout()
    dir_name = os.path.dirname(output_png)
    os.makedirs(dir_name, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return


def make_histogram_anime(
        csv_path: str,
        shp_path: str,
        output_mp4: str,
        dist_range: tuple = (0, 30),
        distance_bin: float = 0.5,
        log_scale: bool = True
):
    logger.info(f"Reading trajectory CSV from {csv_path}")
    trj_df = read_trajectory_csv(csv_path)

    logger.info(f"Reading center point from {shp_path}")
    center_geom = gpd.read_file(shp_path).geometry[0]

    logger.info(f"Calculating histogram with distance")
    hist_df, x_arr = histogram_with_distance(
        trj_df=trj_df,
        center_point=center_geom,
        dist_range=dist_range,
        distance_bin=distance_bin,
        time_bin=1,
        log_scale=log_scale
    )

    logger.info(f"Output movie to {output_mp4}")
    # output_movie(
    #     hist_df=hist_df,
    #     x_arr=x_arr,
    #     output_mp4=output_mp4
    # )
    boxplot_png = output_mp4.replace('.mp4', '_boxplot.png')
    logger.info(f"Output boxplot to {boxplot_png}")
    output_boxplot(
        hist_df=hist_df,
        x_arr=x_arr,
        output_png=boxplot_png,
        log_scale=log_scale
    )
    mean_hist_png = output_mp4.replace('.mp4', '_mean_histogram.png')
    logger.info(f"Output mean histogram to {mean_hist_png}")
    output_mean_histogram(
        hist_df=hist_df,
        x_arr=x_arr,
        output_png=mean_hist_png,
        log_scale=log_scale
    )
    return


def make_index_histogram(
        csv_path: str,
        shp_path: str,
        output_folder: str,
        time_bin: int = 1,
        threshold: float = 3.0
):
    logger.info(f"Reading trajectory CSV from {csv_path}")
    trj_df = read_trajectory_csv(csv_path)

    logger.info(f"Reading center point from {shp_path}")
    center_geom = gpd.read_file(shp_path).geometry[0]

    logger.info(f"Calculating index histogram")
    hist_df = index_histogram(
        trj_df=trj_df,
        center_point=center_geom,
        time_bin=time_bin,
        threshold=threshold
    )
    return


def output_simple_density(
        hist_df: pd.DataFrame,
        output_png: str,
        column_list: list = None,
        log_scale: bool = True,
        x_lim: tuple = None,
        y_lim: tuple = None,
):
    """target/noiseカウントの同時確率分布と周辺分布を出力する。

    中央に2次元ヒストグラムから求めた同時確率分布のヒートマップと
    観測値の散布図を重ね、上側と右側に各変数の周辺ヒストグラムを描画する。

    Parameters
    ----------
    hist_df : pd.DataFrame
        描画対象の2列を含むDataFrame。
    output_png : str
        出力PNGファイルのパス。
    column_list : list[str], optional
        [x軸の列名, y軸の列名]。既定値は
        ['noise_count', 'target_count']。
    log_scale : bool
        Trueの場合、各カウントにlog1p変換を適用して描画する。
    x_lim : tuple[float, float], optional
        X軸の表示範囲。log_scale=Trueの場合も実際のカウント値で指定する。
    y_lim : tuple[float, float], optional
        Y軸の表示範囲。log_scale=Trueの場合も実際のカウント値で指定する。

    Returns
    -------
    None
    """
    if column_list is None:
        column_list = ['noise_count', 'target_count']

    if len(column_list) != 2:
        raise ValueError("column_list must contain exactly two column names")

    x_col, y_col = column_list
    missing_columns = [c for c in column_list if c not in hist_df.columns]
    if missing_columns:
        raise KeyError(f"Columns not found in hist_df: {missing_columns}")

    plot_df = hist_df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if plot_df.empty:
        raise ValueError("No finite data are available for output_simple_density")

    x_raw = plot_df[x_col].to_numpy(dtype=float)
    y_raw = plot_df[y_col].to_numpy(dtype=float)
    if np.any(x_raw < 0) or np.any(y_raw < 0):
        raise ValueError("Count values must be non-negative")

    def validate_limit(limit, name):
        if limit is None:
            return None
        if len(limit) != 2:
            raise ValueError(f"{name} must contain exactly two values")
        lower, upper = map(float, limit)
        if lower < 0 or upper <= lower:
            raise ValueError(f"{name} must satisfy 0 <= min < max")
        return lower, upper

    x_lim = validate_limit(x_lim, 'xlim')
    y_lim = validate_limit(y_lim, 'ylim')

    if log_scale:
        x = np.log1p(x_raw)
        y = np.log1p(y_raw)
        plot_x_lim = tuple(np.log1p(x_lim)) if x_lim is not None else None
        plot_y_lim = tuple(np.log1p(y_lim)) if y_lim is not None else None
    else:
        x = x_raw
        y = y_raw
        plot_x_lim = x_lim
        plot_y_lim = y_lim

    # データ数に応じてbin数を決める。極端に細かくならないよう上限を設ける。
    n_samples = len(plot_df)
    n_bins = int(np.clip(np.sqrt(n_samples), 10, 50))

    # 値が一定の場合でもhistogram2dが計算できるよう、描画範囲を少し広げる。
    def make_range(values: np.ndarray):
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        if np.isclose(value_min, value_max):
            margin = max(abs(value_min) * 0.05, 0.5)
            return value_min - margin, value_max + margin
        margin = (value_max - value_min) * 0.03
        return value_min - margin, value_max + margin

    x_range = plot_x_lim if plot_x_lim is not None else make_range(x)
    y_range = plot_y_lim if plot_y_lim is not None else make_range(y)

    joint_count, x_edges, y_edges = np.histogram2d(
        x,
        y,
        bins=n_bins,
        range=[x_range, y_range],
    )
    total_count = joint_count.sum()
    if total_count == 0:
        raise ValueError(
            'No observations fall within the specified xlim and ylim'
        )
    joint_probability = joint_count / total_count

    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        left=0.10,
        right=0.90,
        bottom=0.10,
        top=0.90,
        wspace=0.05,
        hspace=0.05,
    )
    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax_joint = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax_joint)

    mesh = ax_joint.pcolormesh(
        x_edges,
        y_edges,
        joint_probability.T,
        shading='auto',
        cmap='viridis',
    )
    ax_joint.scatter(
        x,
        y,
        s=12,
        alpha=0.25,
        edgecolors='none',
        label='Observations',
    )

    # 周辺分布も総和が1になるよう、各サンプルに1/Nの重みを付ける。
    weights = np.full(n_samples, 1.0 / n_samples)
    ax_hist_x.hist(
        x,
        bins=x_edges,
        weights=weights,
        edgecolor='black',
        linewidth=0.4,
    )
    ax_hist_y.hist(
        y,
        bins=y_edges,
        weights=weights,
        orientation='horizontal',
        edgecolor='black',
        linewidth=0.4,
    )

    ax_joint.set_xlabel(x_col)
    ax_joint.set_ylabel(y_col)

    if plot_x_lim is not None:
        ax_joint.set_xlim(plot_x_lim)
    if plot_y_lim is not None:
        ax_joint.set_ylim(plot_y_lim)

    if log_scale:
        def set_count_ticks(axis, raw_limit, observed_values, is_x_axis):
            if raw_limit is None:
                lower = 0.0
                upper = float(np.max(observed_values))
            else:
                lower, upper = raw_limit

            # 表示範囲が狭い場合は、実カウント値を1刻みで表示する。
            # 広い場合は、ラベルが過密にならないよう代表値だけ表示する。
            if upper - lower <= 20:
                ticks_raw = np.arange(
                    np.ceil(lower),
                    np.floor(upper) + 1,
                    dtype=float,
                )
            else:
                candidates = np.asarray(DISTANCE_CAND, dtype=float)
                ticks_raw = candidates[(candidates >= lower) & (candidates <= upper)]

            # xlim/ylimの端点は表示範囲としてのみ使用し、目盛りには強制追加しない。
            # これにより39.2や6.2のような端数の最大値が目盛りに現れるのを防ぐ。
            ticks_raw = np.unique(ticks_raw)

            # 候補が1つもない場合だけ、表示範囲内の整数値を補助的に使う。
            if len(ticks_raw) == 0:
                first_tick = int(np.ceil(lower))
                last_tick = int(np.floor(upper))
                if first_tick <= last_tick:
                    ticks_raw = np.arange(first_tick, last_tick + 1, dtype=float)
                else:
                    ticks_raw = np.array([lower], dtype=float)

            ticks_plot = np.log1p(ticks_raw)
            labels = [f'{v:g}' for v in ticks_raw]

            if is_x_axis:
                axis.set_xticks(ticks_plot)
                axis.set_xticklabels(labels)
            else:
                axis.set_yticks(ticks_plot)
                axis.set_yticklabels(labels)

        set_count_ticks(ax_joint, x_lim, x_raw, True)
        set_count_ticks(ax_joint, y_lim, y_raw, False)
    ax_joint.grid(True, alpha=0.2)

    ax_hist_x.set_ylabel('Probability')
    ax_hist_y.set_xlabel('Probability')
    ax_hist_x.tick_params(axis='x', labelbottom=False)
    ax_hist_y.tick_params(axis='y', labelleft=False)

    colorbar = fig.colorbar(mesh, ax=[ax_joint, ax_hist_x, ax_hist_y], pad=0.02)
    colorbar.set_label('Joint probability')
    fig.suptitle(f'Joint probability distribution: {x_col} vs {y_col}')

    output_dir = os.path.dirname(output_png)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return


def make_simple_density(
        csv_path: str,
        shp_path: str,
        output_folder: str,
        time_bin: int = 1,
        threshold: float = 3.0,
        log_scale: bool = True,
):
    logger.info(f"Reading trajectory CSV from {csv_path}")
    trj_df = read_trajectory_csv(csv_path)

    logger.info(f"Reading center point from {shp_path}")
    center_geom = gpd.read_file(shp_path).geometry[0]

    logger.info(f"Calculating simple histogram")
    hist_df = histogram_split_by_threshold(
        trj_df=trj_df,
        center_point=center_geom,
        threshold=threshold,
        time_bin=time_bin
    )

    logger.info(f'Output simple histogram to {output_folder}')
    output_simple_density(
        hist_df=hist_df,
        output_png=os.path.join(output_folder, 'simple_density.png'),
        log_scale=log_scale
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Histogram')
    parser.add_argument('-opt', '--option', type=str, default='anime',
                        choices=['anime', 'index', 'simple'],)
    parser.add_argument('-i', '--input_csv', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('-s', '--center_shp', type=str, required=True,
                        help='Input SHP file path for center point')
    parser.add_argument('-o', '--output_mp4', type=str, default=None,
                        help='Output MP4 file path')
    parser.add_argument('-of', '--output_folder', type=str, default=None,)
    parser.add_argument('-l', '--log1p', type=str, default='True', choices=['True', 'False'],
                        help='Apply log1p to histogram counts')
    parser.add_argument('-dt', '--delta_t', type=float, default=1.0,
                        help='Time interval between points (seconds)')
    parser.add_argument('-dr', '--dist_range', type=float, nargs=2, default=(0, 30),
                        help='Distance range for histogram (min max)')
    parser.add_argument('-db', '--distance_bin', type=float, default=0.5,
                        help='Distance bin width for histogram')
    parser.add_argument('-tb', '--time_bin', type=int, default=5,
                        help='Time bin width for histogram (seconds)')
    parser.add_argument('-th', '--threshold', type=float, default=3.0,)
    args = parser.parse_args()

    if args.option == 'anime':
        make_histogram_anime(
            csv_path=args.input_csv,
            shp_path=args.center_shp,
            dist_range=args.dist_range,
            output_mp4=args.output_mp4,
            distance_bin=args.distance_bin,
            log_scale=(args.log1p == 'True'),
        )
    elif args.option == 'index':
        pass
    elif args.option == 'simple':
        make_simple_density(
            csv_path=args.input_csv,
            shp_path=args.center_shp,
            output_folder=args.output_folder,
            time_bin=args.time_bin,
            threshold=args.threshold,
            log_scale=(args.log1p == 'True'),
        )
    else:
        pass
