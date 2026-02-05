#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -63.6, -12.5),  # Chin
        (-43.3, 32.7, -26.0),  # Left eye outer corner
        (43.3, 32.7, -26.0),  # Right eye outer corner
        (-28.9, -28.9, -24.1),  # Left mouth corner
        (28.9, -28.9, -24.1),  # Right mouth corner
    ],
    dtype=np.float64,
)


LANDMARK_INDEXES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor head pose and face position from the mac camera. "
            "Trigger an alert if the gaze/body is outside the screen range "
            "for a continuous duration."
        )
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--offscreen-seconds", type=float, default=10.0)
    parser.add_argument("--alert-cooldown", type=float, default=10.0)
    parser.add_argument("--yaw-threshold", type=float, default=25.0)
    parser.add_argument("--pitch-threshold", type=float, default=20.0)
    parser.add_argument("--min-face-area", type=float, default=0.06)
    parser.add_argument("--max-face-area", type=float, default=0.50)
    parser.add_argument("--max-center-offset", type=float, default=0.22)
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def estimate_head_pose(
    landmarks: list,
    image_width: int,
    image_height: int,
) -> Optional[Tuple[float, float, float]]:
    image_points = np.array(
        [
            (
                landmarks[LANDMARK_INDEXES["nose_tip"]].x * image_width,
                landmarks[LANDMARK_INDEXES["nose_tip"]].y * image_height,
            ),
            (
                landmarks[LANDMARK_INDEXES["chin"]].x * image_width,
                landmarks[LANDMARK_INDEXES["chin"]].y * image_height,
            ),
            (
                landmarks[LANDMARK_INDEXES["left_eye_outer"]].x * image_width,
                landmarks[LANDMARK_INDEXES["left_eye_outer"]].y * image_height,
            ),
            (
                landmarks[LANDMARK_INDEXES["right_eye_outer"]].x * image_width,
                landmarks[LANDMARK_INDEXES["right_eye_outer"]].y * image_height,
            ),
            (
                landmarks[LANDMARK_INDEXES["mouth_left"]].x * image_width,
                landmarks[LANDMARK_INDEXES["mouth_left"]].y * image_height,
            ),
            (
                landmarks[LANDMARK_INDEXES["mouth_right"]].x * image_width,
                landmarks[LANDMARK_INDEXES["mouth_right"]].y * image_height,
            ),
        ],
        dtype=np.float64,
    )

    focal_length = image_width
    center = (image_width / 2, image_height / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        roll = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    else:
        pitch = math.degrees(math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))
        yaw = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        roll = 0.0

    return yaw, pitch, roll


def compute_face_metrics(
    landmarks: list, image_width: int, image_height: int
) -> Tuple[float, float, Tuple[int, int, int, int]]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    min_x = max(int(min(xs) * image_width), 0)
    max_x = min(int(max(xs) * image_width), image_width - 1)
    min_y = max(int(min(ys) * image_height), 0)
    max_y = min(int(max(ys) * image_height), image_height - 1)

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    face_area_ratio = (bbox_width * bbox_height) / (image_width * image_height)

    face_center_x = (min_x + max_x) / 2
    face_center_y = (min_y + max_y) / 2
    center_offset = math.sqrt(
        ((face_center_x - image_width / 2) / image_width) ** 2
        + ((face_center_y - image_height / 2) / image_height) ** 2
    )
    return face_area_ratio, center_offset, (min_x, min_y, max_x, max_y)


def trigger_alert() -> None:
    message = "Please look at the screen"
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "Screen Monitor"',
                ]
            )
            subprocess.Popen(["say", "Please look at the screen"])
        except OSError:
            print("\a", flush=True)
    else:
        print("\a", flush=True)
        sys.stderr.write("ALERT: " + message + "\n")


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Failed to open camera.", file=sys.stderr)
        return 1

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    offscreen_started = None
    last_alert_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.", file=sys.stderr)
                break

            image_height, image_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = face_mesh.process(rgb_frame)

            out_of_range = False
            reason = ""
            yaw = pitch = roll = None
            face_area_ratio = center_offset = None
            bbox = None

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                pose = estimate_head_pose(landmarks, image_width, image_height)
                if pose is None:
                    out_of_range = True
                    reason = "pose"
                else:
                    yaw, pitch, roll = pose
                    face_area_ratio, center_offset, bbox = compute_face_metrics(
                        landmarks, image_width, image_height
                    )
                    gaze_out = (
                        abs(yaw) > args.yaw_threshold
                        or abs(pitch) > args.pitch_threshold
                    )
                    size_out = (
                        face_area_ratio < args.min_face_area
                        or face_area_ratio > args.max_face_area
                    )
                    center_out = center_offset > args.max_center_offset

                    out_of_range = gaze_out or size_out or center_out
                    reason_parts = []
                    if gaze_out:
                        reason_parts.append("gaze")
                    if size_out:
                        reason_parts.append("distance")
                    if center_out:
                        reason_parts.append("position")
                    reason = ",".join(reason_parts)
            else:
                out_of_range = True
                reason = "no face"

            now = time.monotonic()
            if out_of_range:
                if offscreen_started is None:
                    offscreen_started = now
                offscreen_duration = now - offscreen_started
                if (
                    offscreen_duration >= args.offscreen_seconds
                    and now - last_alert_time >= args.alert_cooldown
                ):
                    trigger_alert()
                    last_alert_time = now
            else:
                offscreen_started = None
                offscreen_duration = 0.0

            if not args.no_preview:
                status_color = (0, 255, 0) if not out_of_range else (0, 0, 255)
                status_text = (
                    "OK" if not out_of_range else f"OUT {offscreen_duration:0.1f}s"
                )
                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )
                if reason:
                    cv2.putText(
                        frame,
                        f"Reason: {reason}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        status_color,
                        2,
                    )
                if yaw is not None and pitch is not None:
                    cv2.putText(
                        frame,
                        f"Yaw: {yaw:0.1f} Pitch: {pitch:0.1f} Roll: {roll:0.1f}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                if face_area_ratio is not None and center_offset is not None:
                    cv2.putText(
                        frame,
                        f"Face area: {face_area_ratio:0.3f} Center off: {center_offset:0.3f}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                if bbox is not None:
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (255, 255, 0),
                        2,
                    )

                cv2.imshow("Screen Monitor", frame)
                key = cv2.waitKey(1)
                if key in (ord("q"), 27):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        face_mesh.close()
        if not args.no_preview:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
