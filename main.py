#!/usr/bin/env python3
"""
Hand Trigger Recorder (Global, cross-platform)
- When a hand is stably visible, recording starts (writing frames to an open file).
- When the hand is stably absent, it waits; on the next stable reappearance, recording PAUSES (file stays open).
- After the hand is fully absent again and reappears, it ARMs to start a NEW recording file next time.
- The preview window remains open; status is shown at top-left.
"""

from __future__ import annotations
import os
import cv2
import time
import argparse
from pathlib import Path
from datetime import datetime
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hand-triggered video recorder (pause/resume without closing file)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--flip", action="store_true", default=True, help="Mirror (selfie) view (default: on)")
    p.add_argument("--min-det", type=float, default=0.7, help="MediaPipe min_detection_confidence")
    p.add_argument("--min-trk", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
    p.add_argument("--present-stable", type=float, default=0.4, help="Stable time (s) to accept 'hand present'")
    p.add_argument("--absent-stable", type=float, default=0.6, help="Stable time (s) to accept 'hand absent'")
    p.add_argument("--fps-fallback", type=int, default=30, help="Fallback FPS if camera does not report one")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <script_dir>/records/YYYY-MM-DD)")
    p.add_argument("--window", type=str, default="Hand Trigger Recorder", help="Window title")
    return p.parse_args()


def script_dir() -> Path:
    return os.path.abspath(__file__) #__file__ should look like this: C:\Users\your_user\Desktop\Hand-Trigger-Recorder


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_output_dir(base: str | None) -> Path:
    if base is None:
        # Daily folder: records/YYYY-MM-DD next to the script
        root = script_dir() / "records" / datetime.now().strftime("%Y-%m-%d")
    else:
        root = Path(base).expanduser().resolve()
    ensure_dir(root)
    return root


def open_writer_safe(out_dir: Path, w: int, h: int, fps: float) -> tuple[cv2.VideoWriter | None, Path | None]:
    """Try MP4 (mp4v), fall back to AVI (XVID) if needed."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    mp4_path = out_dir / f"hand_record_{ts}.mp4"
    wr = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if wr.isOpened():
        print(f"[INFO] MP4 -> {mp4_path}")
        return wr, mp4_path

    avi_path = out_dir / f"hand_record_{ts}.avi"
    wr = cv2.VideoWriter(str(avi_path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if wr.isOpened():
        print(f"[WARN] mp4v not available, falling back to AVI/XVID -> {avi_path}")
        return wr, avi_path

    print("[ERR] VideoWriter could not be opened (neither MP4 nor AVI)")
    return None, None


def put_status(img, text: str, color=(0, 215, 255)) -> None:
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    # CAP_DSHOW reduces latency on Windows; harmless elsewhere (OpenCV ignores it)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = float(args.fps_fallback)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    out_dir = make_output_dir(args.out_dir)

    window = args.window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"[INFO] Camera: {w}x{h} @ ~{fps:.1f} fps")
    print(f"[INFO] Output directory: {out_dir}")
    print("[INFO] Press ESC to quit.")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils

    # State machine:
    # ARMED -> (present stable) -> RECORDING_WAIT_LOSS
    # RECORDING_WAIT_LOSS -> (absent stable) -> RECORDING_WAIT_REAPPEAR
    # RECORDING_WAIT_REAPPEAR -> (present stable) -> STOPPED_REARM_WAIT_ABSENCE (writing pauses; file stays open)
    # STOPPED_REARM_WAIT_ABSENCE -> (absent stable) -> ARMED
    state = "ARMED"
    present_since = None
    absent_since = None

    writer = None
    out_path: Path | None = None
    writing = False  # True: writer.write(frame) is active

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=args.min_trk,
        model_complexity=1
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERR] Failed to grab frame")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            hand_present = res.multi_hand_landmarks is not None

            now = time.time()
            if hand_present:
                present_since = now if present_since is None else present_since
                absent_since = None
            else:
                absent_since = now if absent_since is None else absent_since
                present_since = None

            # --- STATE LOGIC ---
            if state == "ARMED":
                put_status(frame, "STOPPED", (0, 215, 255))
                if present_since and (now - present_since) >= args.present_stable:
                    writer, out_path = open_writer_safe(out_dir, w, h, fps)
                    if writer is None:
                        break
                    writing = True
                    print(f"[INFO] Recording started: {out_path}")
                    state = "RECORDING_WAIT_LOSS"

            elif state == "RECORDING_WAIT_LOSS":
                put_status(frame, "STARTED", (0, 0, 255))
                if absent_since and (now - absent_since) >= args.absent_stable:
                    state = "RECORDING_WAIT_REAPPEAR"

            elif state == "RECORDING_WAIT_REAPPEAR":
                put_status(frame, "STARTED (will PAUSE when hand reappears)", (0, 0, 255))
                if present_since and (now - present_since) >= args.present_stable:
                    writing = False  # keep file open; just pause writing
                    print(f"[INFO] Recording PAUSED (file open): {out_path}")
                    state = "STOPPED_REARM_WAIT_ABSENCE"

            elif state == "STOPPED_REARM_WAIT_ABSENCE":
                put_status(frame, "STOPPED (to start again: remove hand, then show)", (0, 215, 255))
                if absent_since and (now - absent_since) >= args.absent_stable:
                    state = "ARMED"

            # Optional: draw landmarks for visual feedback
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Write frames only while 'writing' is True
            if writer is not None and writing:
                writer.write(frame)

            cv2.imshow(window, frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    # Cleanup
    if writer is not None:
        writer.release()
        print(f"[INFO] File closed: {out_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
