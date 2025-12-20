# annotate_feet_with_audio.py
import cv2
import mediapipe as mp
import argparse
import csv
import os
import subprocess
import shutil
import tempfile
import sys

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

def ema(prev, new, beta):
    if prev is None or beta <= 0:
        return new
    return (1 - beta) * prev + beta * new

def run_cmd(cmd):
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr.decode(errors="ignore") if e.stderr else "")
        return False
    except FileNotFoundError:
        # ffmpeg/ffprobeが見つからない場合
        return False

def get_rotation_deg(input_path):
    """
    FFprobeで回転メタデータ(0/90/180/270)を取得。無ければ0。
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream_tags=rotate",
        "-of", "default=nk=1:nw=1",
        input_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        txt = out.decode().strip()
        if txt == "":
            return 0
        deg = int(txt) % 360
        if deg in (0, 90, 180, 270):
            return deg
        return 0
    except Exception:
        return 0

def rotate_frame_if_needed(frame, rot):
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def main():
    p = argparse.ArgumentParser(description="Detect ankle positions, overlay markers, keep audio and size.")
    p.add_argument("--input", required=True, help="Input MP4 path")
    p.add_argument("--output", required=True, help="Output MP4 path")
    p.add_argument("--csv", default=None, help="Optional CSV path to save ankle coordinates per frame")
    p.add_argument("--alpha", type=float, default=0.7, help="Overlay alpha (0-1)")
    p.add_argument("--smooth", type=float, default=0.3, help="EMA smoothing factor (0-1, 0 disables)")
    p.add_argument("--min_vis", type=float, default=0.5, help="Minimum visibility/confidence to draw")
    p.add_argument("--codec", default="mp4v", help="FourCC for temp video (e.g., mp4v, avc1)")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # 回転メタデータを取得（表示サイズ/向きを維持するため）
    rotation_deg = get_rotation_deg(args.input)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    # OpenCVが返す幅高（回転未適用のことがある）
    src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 表示上の最終サイズ（回転が90/270のときは縦横が入れ替わる）
    if rotation_deg in (90, 270):
        out_w, out_h = src_h, src_w
    else:
        out_w, out_h = src_w, src_h

    fourcc = cv2.VideoWriter_fourcc(*args.codec)

    # 一時ファイル置き場
    workdir = tempfile.mkdtemp(prefix="annotate_")
    temp_video = os.path.join(workdir, "temp_video.mp4")
    temp_audio = os.path.join(workdir, "temp_audio.m4a")

    out = cv2.VideoWriter(temp_video, fourcc, fps, (out_w, out_h))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try another --codec (e.g., avc1).")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx", "left_x", "left_y", "left_vis", "right_x", "right_y", "right_vis"])

    l_prev = None
    r_prev = None

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 回転メタデータを実画に反映（表示通りのサイズで書き出す）
            frame = rotate_frame_if_needed(frame, rotation_deg)

            # 推論はRGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            overlay = frame.copy()
            drew_any = False

            lxy = None
            rxy = None
            lvis = 0.0
            rvis = 0.0

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                l = lm[LEFT_ANKLE]
                r = lm[RIGHT_ANKLE]

                lx, ly = int(l.x * out_w), int(l.y * out_h)
                rx, ry = int(r.x * out_w), int(r.y * out_h)
                lvis = float(getattr(l, "visibility", 0.0))
                rvis = float(getattr(r, "visibility", 0.0))

                # EMA smoothing
                if 0 < args.smooth <= 1:
                    l_prev = (ema(l_prev[0] if l_prev else None, lx, args.smooth),
                              ema(l_prev[1] if l_prev else None, ly, args.smooth)) if lvis >= args.min_vis else l_prev
                    r_prev = (ema(r_prev[0] if r_prev else None, rx, args.smooth),
                              ema(r_prev[1] if r_prev else None, ry, args.smooth)) if rvis >= args.min_vis else r_prev
                    if lvis >= args.min_vis and l_prev is not None:
                        lx, ly = int(l_prev[0]), int(l_prev[1])
                    if rvis >= args.min_vis and r_prev is not None:
                        rx, ry = int(r_prev[0]), int(r_prev[1])

                lxy = (lx, ly)
                rxy = (rx, ry)

                radius = max(6, min(out_w, out_h) // 120)
                thickness = max(2, radius // 3)

                if lvis >= args.min_vis:
                    cv2.circle(overlay, lxy, radius, (255, 120, 0), -1)      # Left: blue-ish
                    cv2.circle(overlay, lxy, radius, (255, 255, 255), thickness)
                    cv2.putText(overlay, "L", (lxy[0] + radius + 4, lxy[1] - radius - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    drew_any = True

                if rvis >= args.min_vis:
                    cv2.circle(overlay, rxy, radius, (0, 80, 255), -1)       # Right: red-ish
                    cv2.circle(overlay, rxy, radius, (255, 255, 255), thickness)
                    cv2.putText(overlay, "R", (rxy[0] + radius + 4, rxy[1] - radius - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    drew_any = True

            if drew_any:
                frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)

            # 凡例
            cv2.rectangle(frame, (10, 10), (190, 50), (0, 0, 0), -1)
            cv2.putText(frame, "L: Left ankle", (18, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "R: Right ankle", (18, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 2, cv2.LINE_AA)

            # フレームサイズが変わらないように固定
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

            out.write(frame)

            if csv_writer is not None:
                row = [frame_idx]
                if lxy:
                    row += [lxy[0], lxy[1], f"{lvis:.3f}"]
                else:
                    row += ["", "", ""]
                if rxy:
                    row += [rxy[0], rxy[1], f"{rvis:.3f}"]
                else:
                    row += ["", "", ""]
                csv_writer.writerow(row)

            frame_idx += 1

    finally:
        cap.release()
        out.release()
        pose.close()
        if csv_file:
            csv_file.close()

    # ここから音声を再ミックス
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    ffprobe_ok = shutil.which("ffprobe") is not None
    audio_mixed = False

    if ffmpeg_ok:
        # まず元動画から音声抽出（可逆コピーを優先、失敗時はAAC再エンコード）
        # 可逆コピー
        if run_cmd(["ffmpeg", "-y", "-i", args.input, "-vn", "-c:a", "copy", temp_audio]):
            pass
        else:
            # 再エンコード（AAC 192kbps）
            run_cmd(["ffmpeg", "-y", "-i", args.input, "-vn", "-c:a", "aac", "-b:a", "192k", temp_audio])

        # 回転メタデータを継承（OpenCVで実画を回転済みなら0でも問題なし。ここはできるだけ元を踏襲）
        rotate_meta = str(rotation_deg if rotation_deg in (0,90,180,270) else 0)

        # 映像は temp_video をコピー、音声は抽出したものをコピーで最短に合わせる
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,
            "-i", temp_audio,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy",
            "-shortest"
        ]
        # 可能なら回転メタデータを設定
        if ffprobe_ok:
            mux_cmd += ["-metadata:s:v:0", f"rotate={rotate_meta}"]

        mux_cmd += [args.output]
        audio_mixed = run_cmd(mux_cmd)

    if not ffmpeg_ok or not audio_mixed:
        # FFmpegが無い or ミックス失敗 → 映像のみのファイルをそのまま出力にコピー
        # （サイズは維持されますが、音声は入りません）
        shutil.copyfile(temp_video, args.output)
        if not ffmpeg_ok:
            print("Warning: FFmpeg/FFprobe not found. Output has NO audio.")
        else:
            print("Warning: Failed to mux audio. Output has NO audio.")

    # 後片付け
    try:
        shutil.rmtree(workdir)
    except Exception:
        pass

    print(f"Done. Saved video to: {args.output}")
    if args.csv:
        print(f"Saved ankle coordinates to: {args.csv}")

if __name__ == "__main__":
    main()
