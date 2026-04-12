import os.path
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from common.logger import get_logger

logger = get_logger('tools.create_heatmap_mp4')


def parse_coordinates(columns, prefix='target_count_'):
    """
    カラム名 prefix + x{i}_y{j} から (i, j) を抽出
    """
    coords = {}
    pattern = re.compile(rf"{re.escape(prefix)}x(-?\d+)_y(-?\d+)$")

    for col in columns:
        match = pattern.fullmatch(col)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            coords[col] = (x, y)

    return coords


def build_grid_mapping(coords):
    """
    座標からグリッドのサイズとインデックス対応を作る
    """
    xs = sorted(set(x for x, _ in coords.values()))
    ys = sorted(set(y for _, y in coords.values()))

    x_to_idx = {x: i for i, x in enumerate(xs)}
    y_to_idx = {y: i for i, y in enumerate(ys)}

    return xs, ys, x_to_idx, y_to_idx


def make_heatmap_mp4_auto(
    df: pd.DataFrame,
    output_path: str,
    fps: float = 1.0,
    cmap: str = "hot",
    v_min=None,
    v_max=None,
    title: str = "Heatmap Animation"
):
    # 1. 座標を抽出
    coords = parse_coordinates(df.columns)
    col_list = [col for col in coords.keys()]
    df = df[col_list]
    logger.info(f"Parsed coordinates: {len(coords)} columns found.")

    if len(coords) == 0:
        raise ValueError("対象となる x{i}_y{j} 形式のカラムが見つかりません。")

    if fps <= 0:
        raise ValueError("fps must be greater than 0.")

    # 2. グリッド構造を構築
    xs, ys, x_to_idx, y_to_idx = build_grid_mapping(coords)
    nx, ny = len(xs), len(ys)

    # 3. 色スケール
    if v_min is None:
        v_min = float(df.min().min())
    if v_max is None:
        v_max = float(df.max().max())
    logger.info(f"Color scale: vmin={v_min}, vmax={v_max}")

    # 4. 初期フレーム
    def row_to_grid(row):
        grid = np.zeros((ny, nx))  # (y, x)
        for col, value in row.items():
            if col not in coords:
                continue
            x, y = coords[col]
            grid[y_to_idx[y], x_to_idx[x]] = value
        return grid

    fig, ax = plt.subplots(figsize=(5, 5))

    first_grid = row_to_grid(df.iloc[0])
    im = ax.imshow(first_grid, cmap=cmap, vmin=v_min, vmax=v_max)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    title_text = ax.set_title(f"{title}\nTime = 0 s")

    # 軸ラベルを座標値にする
    ax.set_xticks(range(nx))
    ax.set_xticklabels(xs)
    ax.set_yticks(range(ny))
    ax.set_yticklabels(ys)

    # グリッド線
    ax.set_xticks(np.arange(-0.5, nx, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ny, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)

    def update(frame_idx):
        grid = row_to_grid(df.iloc[frame_idx])
        im.set_array(grid)
        title_text.set_text(f"{title}\nTime = {frame_idx} s")
        return [im, title_text]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=1000 / fps,
        blit=True
    )

    writer = FFMpegWriter(fps=fps)
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    anim.save(output_path, writer=writer)
    plt.close(fig)

    logger.info(f"Saved: {output_path}")


# ===== 使用例 =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str)
    parser.add_argument('-o', '--output-file', type=str)
    parser.add_argument('--fps', type=float, default=1.0, help='Frames per second of output mp4')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap for the heatmap')
    parser.add_argument('--vmin', type=float, default=None, help='Minimum value for color scale')
    parser.add_argument('--vmax', type=float, default=None, help='Maximum value for color scale')
    parser.add_argument('--title', type=str, default='Heatmap Animation', help='Title for the heatmap')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    make_heatmap_mp4_auto(
        df=df,
        output_path=args.output_file,
        fps=args.fps,
        cmap=args.cmap,
        v_min=args.vmin,
        v_max=args.vmax,
        title=args.title
    )