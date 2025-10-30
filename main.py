#!/usr/bin/env python3
"""
Hand Trigger Recorder
- El stabil görününce kayıt başlar.
- El stabil kaybolunca bekler; tekrar stabil görünürse kayıt DURUR (dosya açık kalır).
- El tamamen yok olduktan sonra bir daha görünürse yeni kayıt başlatmak için tekrar ARMED durumuna döner.
- Pencere kapanmaz; sol üstte durum görünür.
"""

from __future__ import annotations
import os
import cv2
import time
import argparse
from datetime import datetime
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="El tetiklemeli video kaydedici")
    p.add_argument("--camera", type=int, default=0, help="Kamera index (varsayılan: 0)")
    p.add_argument("--flip", action="store_true", default=True, help="Ayna görünümü (varsayılan: açık)")
    p.add_argument("--min-det", type=float, default=0.7, help="MediaPipe min_detection_confidence")
    p.add_argument("--min-trk", type=float, default=0.5, help="MediaPipe min_tracking_confidence")
    p.add_argument("--present-stable", type=float, default=0.4, help="'el var' stabil süresi sn")
    p.add_argument("--absent-stable", type=float, default=0.6, help="'el yok' stabil süresi sn")
    p.add_argument("--fps-fallback", type=int, default=30, help="Kameradan FPS alınamazsa kullanılacak")
    p.add_argument("--out-dir", type=str, default=None, help="Kayıtların konacağı klasör (varsayılan: script dizini/records/YYYY-MM-DD)")
    p.add_argument("--window", type=str, default="Hand Trigger Recorder", help="Pencere adı")
    return p.parse_args()


def script_dir() -> str:
    return os.path.abspath("C:\\Users\\beyaz\\Desktop\\hand_scan")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_output_dir(base: str | None) -> str:
    if base is None:
        # Günlük klasör: records/YYYY-MM-DD
        root = os.path.join(script_dir(), "records", datetime.now().strftime("%Y-%m-%d"))
    else:
        root = os.path.abspath(base)
    ensure_dir(root)
    return root


def open_writer_safe(out_dir: str, w: int, h: int, fps: float) -> tuple[cv2.VideoWriter | None, str | None]:
    """Önce MP4 (mp4v), olmazsa AVI (XVID)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mp4_path = os.path.join(out_dir, f"hand_record_{ts}.mp4")
    wr = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if wr.isOpened():
        print(f"[INFO] MP4 -> {mp4_path}")
        return wr, mp4_path

    avi_path = os.path.join(out_dir, f"hand_record_{ts}.avi")
    wr = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if wr.isOpened():
        print(f"[WARN] mp4v açılamadı, AVI/XVID -> {avi_path}")
        return wr, avi_path

    print("[ERR] VideoWriter açılamadı (ne MP4 ne AVI)")
    return None, None


def put_status(img, text: str, color=(0, 215, 255)) -> None:
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamadı")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = float(args.fps_fallback)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    out_dir = make_output_dir(args.out_dir)

    window = args.window
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print(f"[INFO] Kamera: {w}x{h} @ ~{fps:.1f} fps")
    print(f"[INFO] Çıkış klasörü: {out_dir}")
    print("[INFO] ESC ile çıkabilirsiniz.")

    mp_hands = mp.solutions.hands
    drawer = mp.solutions.drawing_utils

    # Durum makinesi:
    # ARMED -> (el var stabil) -> RECORDING_WAIT_LOSS
    # RECORDING_WAIT_LOSS -> (el yok stabil) -> RECORDING_WAIT_REAPPEAR
    # RECORDING_WAIT_REAPPEAR -> (el var stabil) -> STOPPED_REARM_WAIT_ABSENCE (yazma durur)
    # STOPPED_REARM_WAIT_ABSENCE -> (el yok stabil) -> ARMED
    state = "ARMED"
    present_since = None
    absent_since = None

    writer = None
    out_path = None
    writing = False  # True: writer.write(frame) aktif

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
                print("[ERR] Kare alınamadı")
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
                put_status(frame, "DURDU", (0, 215, 255))
                if present_since and (now - present_since) >= args.present_stable:
                    writer, out_path = open_writer_safe(out_dir, w, h, fps)
                    if writer is None:
                        break
                    writing = True
                    print(f"[INFO] Kayıt başladı: {out_path}")
                    state = "RECORDING_WAIT_LOSS"

            elif state == "RECORDING_WAIT_LOSS":
                put_status(frame, "BASLADI", (0, 0, 255))
                if absent_since and (now - absent_since) >= args.absent_stable:
                    state = "RECORDING_WAIT_REAPPEAR"

            elif state == "RECORDING_WAIT_REAPPEAR":
                put_status(frame, "BASLADI (el görünürse DURUR)", (0, 0, 255))
                if present_since and (now - present_since) >= args.present_stable:
                    writing = False  # dosya açık kalır, yazma durur
                    print(f"[INFO] Kayıt DURDU (dosya açık): {out_path}")
                    state = "STOPPED_REARM_WAIT_ABSENCE"

            elif state == "STOPPED_REARM_WAIT_ABSENCE":
                put_status(frame, "DURDU (yeniden başlatmak icin elini CEK, sonra goster)", (0, 215, 255))
                if absent_since and (now - absent_since) >= args.absent_stable:
                    state = "ARMED"

            # Landmark (isteğe bağlı)
            if res.multi_hand_landmarks:
                for lm in res.multi_hand_landmarks:
                    drawer.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # Yazma
            if writer is not None and writing:
                writer.write(frame)

            # Pencere hep açık
            cv2.imshow(window, frame)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC
                break

    # Temizlik
    if writer is not None:
        writer.release()
        print(f"[INFO] Dosya kapandı: {out_path}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
