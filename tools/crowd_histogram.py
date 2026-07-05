import os
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import timedelta
from shapely.geometry import LineString, Point
from shapely.wkt import loads as wkt_loads
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common.logger import get_logger


logger = get_logger('tools.crowd_histogram')

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
    if log_scale:
        hist_df['histogram'] = hist_df['histogram'].apply(lambda x: np.log1p(x))
    return hist_df, x_arr


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

    Returns
    -------
    None
    """
    bin_left = x_arr[:-1]
    bin_width = np.diff(x_arr)

    max_count = hist_df['histogram'].apply(
        lambda h: np.max(h) if len(h) > 0 else 0
    ).max()
    if log_scale:
        max_count = np.expm1(max_count)
    # 目盛り候補
    candidates = np.array([0, 1, 2, 5,
                           10, 20, 50,
                           100, 200, 500,
                           1000])

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        bin_left,
        hist_df.iloc[0]['histogram'],
        width=bin_width,
        align='edge',
        edgecolor='black'
    )

    ax.set_xlim(x_arr[0], x_arr[-1])
    ax.set_ylim(0, max_count * 1.1 if max_count > 0 else 1)
    ax.set_xlabel("Distance from center point")
    ax.set_ylabel("Crowd count" + "Crowd count (log1p scale)" if log_scale else "")
    ax.set_title("Crowd distance histogram")
    major_ticks = np.arange(0, x_arr[-1] + 0.1, 5)

    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(int(x)) for x in major_ticks])
    if log_scale:
        y_ticks = candidates[candidates <= max_count]

        ax.set_yticks(np.log1p(y_ticks))
        ax.set_yticklabels([str(v) for v in y_ticks])

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
        log_scale: bool = True
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

    Returns
    -------
    None
    """
    hist_matrix = np.vstack(hist_df['histogram'].values)

    bin_centers = (x_arr[:-1] + x_arr[1:]) / 2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        hist_matrix,
        positions=bin_centers,
        widths=np.diff(x_arr) * 0.7,
        showfliers=False
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
        candidates = np.array([0, 1, 2, 5,
                               10, 20, 50,
                               100, 200, 500,
                               1000])

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


def make_histogram_anime(
        csv_path: str,
        shp_path: str,
        output_mp4: str,
        dist_range: tuple = (0, 30),
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
        distance_bin=0.5,
        time_bin=1,
        log_scale=log_scale
    )

    logger.info(f"Output movie to {output_mp4}")
    output_movie(
        hist_df=hist_df,
        x_arr=x_arr,
        output_mp4=output_mp4
    )
    boxplot_png = output_mp4.replace('.mp4', '_boxplot.png')
    logger.info(f"Output boxplot to {boxplot_png}")
    output_boxplot(
        hist_df=hist_df,
        x_arr=x_arr,
        output_png=boxplot_png,
        log_scale=log_scale
    )
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crowd Histogram')
    parser.add_argument('-i', '--input_csv', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('-s', '--center_shp', type=str, required=True,
                        help='Input SHP file path for center point')
    parser.add_argument('-o', '--output_mp4', type=str, default=None,
                        help='Output MP4 file path')
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
    args = parser.parse_args()

    make_histogram_anime(
        csv_path=args.input_csv,
        shp_path=args.center_shp,
        dist_range=args.dist_range,
        output_mp4=args.output_mp4,
        log_scale=(args.log1p == 'True'),
    )
