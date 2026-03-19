"""
Run one attention mechanism at a time, or all three simultaneously:
  --mode face | motion | sound | idle | all

'all' mode runs face, motion and sound detection together every step.
  - ALL detected events are logged to the CSV (one row per source per step).
  - Head movement priority: face > motion > sound.
    (NAO turns toward the highest-priority active detection only.)
  - LEDs reflect whichever source is driving the head.

LED Feedback:
  - Red:  Face detected  (or face wins priority in 'all' mode)
  - Blue: Motion detected
  - Green: Sound detected
  - Dim:  No detection / idle
"""

import os
import sys
import time
import json
import random
import math

import csv

import qi
import cv2
import numpy as np
from PIL import Image, ImageDraw

def normalize_angle(angle):
    """
    Wrap angle to [-pi, pi]
    
    Args:
        angle (float): input angle in radians

    Returns: 
        normalized angle in radians
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle+= 2 * math.pi
    return angle

def now_time():
    """
    Returns current time as a float in seconds (with sub-second precision).

    Args: 
        None
    Returns: 
        Current time in seconds (float)
    """
    return time.time()


def clamp(x, lo, hi):
    """
    Clamp x to the range [lo, hi].
    
    Args: 
        x (float): value to clamp
        lo (float): lower bound
        hi (float): upper bound
        
    Returns: 
        Clamped value (float)
    """
    return max(lo, min(hi, x))


def get_script_data_root():
    """
    Return the absolute path to the shared data folder one level above this script.

    Args:
        None

    Returns:
        path (str): Normalised absolute path to ``../data`` relative to this file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(script_dir, "..", "data"))


def safe_get_timestamp(mem, key):
    """
    Retrieve a value and its timestamp from ALMemory without raising on failure.
    Uses ALMemory.getTimestamp which returns (value, sec, usec).

    Args:
        mem: ALMemory proxy object.
        key (str): ALMemory key to query.

    Returns:
        tuple: (value, ts_float) where ts_float is seconds since epoch as a float,
               or (None, None) if the key is missing or an error occurs.
    """
    try:
        result = mem.getTimestamp(key)
        val, sec, usec = result
        ts = float(sec) + float(usec) * 1e-6
        return val, ts
    except Exception:
        return None, None


def ang_center_size_to_px_bbox(alpha, beta, size_x, size_y, img_w, img_h, hfov, vfov):
    """
    Convert NAO ShapeInfo angular centre/size to a pixel bounding box.
    Used for FaceDetected and MovementDetection events whose ShapeInfo
    provides (alpha, beta, sizeX, sizeY) in radians.
    X axis is inverted: negative alpha = right in the robot's camera frame.

    Args:
        alpha (float): Horizontal centre angle in radians (negative = right).
        beta (float): Vertical centre angle in radians (positive = down).
        size_x (float): Angular width of the region in radians.
        size_y (float): Angular height of the region in radians.
        img_w (int): Image width in pixels.
        img_h (int): Image height in pixels.
        hfov (float): Camera horizontal field of view in radians.
        vfov (float): Camera vertical field of view in radians.

    Returns:
        list: Bounding box as [x, y, w, h] in pixel coordinates,
              clamped to the image boundaries.
    """
    # Convert angular position to normalized coordinates [-1, 1]
    norm_x = -alpha / (hfov / 2.0)  # INVERTED: alpha negative = right
    norm_y = beta / (vfov / 2.0)  # beta positive = down

    # Convert to pixel coordinates (origin at top-left)
    cx = (norm_x + 1.0) * (img_w / 2.0)
    cy = (norm_y + 1.0) * (img_h / 2.0)

    # Convert angular size to pixel size
    bw = (size_x / hfov) * img_w
    bh = (size_y / vfov) * img_h

    # Calculate top-left corner
    x = cx - bw / 2.0
    y = cy - bh / 2.0

    # Clamp to image bounds
    x = clamp(x, 0, img_w - 1)
    y = clamp(y, 0, img_h - 1)
    bw = clamp(bw, 1, img_w - x)
    bh = clamp(bh, 1, img_h - y)

    return [int(round(x)), int(round(y)), int(round(bw)), int(round(bh))]

class DetectionLogger(object):
    
    COLUMNS = [
        "wall_time", "elapsed_s", "frame_idx", "source", "detected", "confidence",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "head_yaw", "head_pitch",
        "status",
        "num_detections (faces/cluster)",  # num_faces (face rows) or num_clusters (motion rows)
        "azimuth",  #  all attention modes: horizontal angle to attention source (radians)
        "elevation",  #   all attention modes: vertical angle to attention source (radians)
        "energy (sound)",  # sound only: acoustic energy / loudness proxy
        "head_driver",  # 'all' mode only: which source is currently driving the head
        "extra_json",  # remaining debug/error fields
    ]
    def __init__(self, path, session_name, mode, t_start, min_log_interval_sec=0.5):
        """
        Initialise the logger and write the CSV header with session metadata.
        Bounding box is split into four individual numeric columns so it sorts
        and filters cleanly in Excel / pandas without any string parsing.
        Common extra fields (confidence, azimuth, etc.) are promoted to
        their own columns; anything remaining is kept in extra_json.
        A new row is written only when the detection *state changes* or after at
        least `min_log_interval_sec` seconds — whichever comes first — to keep
        the file compact without losing transitions.

        Args:
            path (str): Full file path where the CSV will be written.
            session_name (str): Human-readable session identifier written to the header.
            mode (str): Attention mode (face | motion | sound | idle | all).
            t_start (float): Session start time in seconds (from time.time()).
            min_log_interval_sec (float): Minimum seconds between rows for the same
                source when the detection state has not changed (default 0.5).
        """

        self.path = path
        self.t_start = t_start
        self.min_log_interval = min_log_interval_sec

        self._last_detected = None
        self._last_source = None
        self._last_log_time = 0.0

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            f.write("# NAO Attention Capture - Detection History\n")
            f.write("# session,{}\n".format(session_name))
            f.write("# mode,{}\n".format(mode))
            f.write("# started,{}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))))
            f.write("#\n")
            writer.writerow(self.COLUMNS)

    def log(self, t_now, frame_idx, source, detected, confidence, bbox_px,
            head_yaw, head_pitch, extra, event_image_path=None):
        """
        Write one CSV row if the detection state changed or the log interval elapsed.

        Args:
            t_now (float): Current wall-clock time in seconds.
            frame_idx (int): Zero-based index of the current video frame.
            source (str): Detection source identifier (face | motion | sound | idle).
            detected (bool): True if a detection target is present this step.
            confidence (float or None): Detection confidence score, or None if unavailable.
            bbox_px (list or None): Bounding box [x, y, w, h] in pixels, or None.
            head_yaw (float): Current head yaw angle in radians.
            head_pitch (float): Current head pitch angle in radians.
            extra (dict or None): Additional fields; recognised keys (status, num_faces,
                num_clusters, azimuth, elevation, energy, head_driver) are promoted to
                dedicated columns; remaining keys are serialised to extra_json.
            event_image_path (str or None): Optional path to a saved event image
                (unused in current implementation, reserved for future use).

        Returns:
            bool: True if a row was written, False if skipped.
        """
        state_changed = (detected != self._last_detected) or (source != self._last_source)
        time_due = (t_now - self._last_log_time) >= self.min_log_interval

        if not (state_changed or time_due):
            return False

        elapsed = t_now - self.t_start
        extra = dict(extra) if extra else {}

        bbox_x = bbox_px[0] if bbox_px else ""
        bbox_y = bbox_px[1] if bbox_px else ""
        bbox_w = bbox_px[2] if bbox_px else ""
        bbox_h = bbox_px[3] if bbox_px else ""

        def pop(key):
            v = extra.pop(key, None)
            return "" if v is None else v

        status      = pop("status")
        num_det     = pop("num_faces")
        if num_det == "":
            num_det = pop("num_clusters")
        else:
            extra.pop("num_clusters", None)  
        azimuth     = pop("azimuth")
        elevation   = pop("elevation")
        energy      = pop("energy")
        head_driver = pop("head_driver")

        # Remove internal/redundant keys that should not appear in extra_json
        for k in ("cur_head", "movement_ts", "propAux", "posCog", "mode", "move_count",
                  "label", "scoreReco", "propMoving"):
            extra.pop(k, None)

        extra_json = json.dumps(extra) if extra else ""

        row = [
            "{:.6f}".format(t_now),
            "{:.3f}".format(elapsed),
            int(frame_idx),
            source,
            "yes" if detected else "no",
            "{:.4f}".format(confidence) if confidence is not None else "",
            bbox_x, bbox_y, bbox_w, bbox_h,
            "{:.4f}".format(head_yaw),
            "{:.4f}".format(head_pitch),
            status,
            num_det,
            azimuth, elevation, energy,
            head_driver,
            extra_json,
        ]

        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        self._last_detected = detected
        self._last_source = source
        self._last_log_time = t_now
        return True

    def write_footer(self, t_end, total_frames):
        """
        Append session summary comment lines to the CSV after capture ends.

        Args:
            t_end (float): Session end time in seconds (from time.time()).
            total_frames (int): Total number of video frames captured.
        """
        with open(self.path, "a", newline="") as f:
            f.write("# ended,{}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end))))
            f.write("# duration_s,{:.1f}\n".format(t_end - self.t_start))
            f.write("# total_frames,{}\n".format(total_frames))


# LED Controller
class LEDController(object):
    """
    Controls NAO's eye LEDs based on attention source.
    Colors:
      - Red (0xFF0000): Face detection
      - Blue (0x0000FF): Motion detection
      - Green (0x00FF00): Sound detection
      - White dim (0x111111): Idle/No detection
    """

    COLOR_FACE = 0xFF0000  # Red
    COLOR_MOTION = 0x0000FF  # Blue
    COLOR_SOUND = 0x00FF00  # Green
    COLOR_IDLE = 0x111111  # Dim white
    COLOR_OFF = 0x000000  # Off

    def __init__(self, session):
        self.leds = session.service("ALLeds")
        self.current_color = None

    def set_face_detected(self):
        """
        Set eye LEDs to red, indicating an active face detection.
        No-op if the LEDs are already showing the face colour.
        """
        if self.current_color != "face":
            self.leds.fadeRGB("FaceLeds", self.COLOR_FACE, 0.1)
            self.current_color = "face"

    def set_motion_detected(self):
        """
        Set eye LEDs to blue, indicating active motion detection.
        No-op if the LEDs are already showing the motion colour.
        """
        if self.current_color != "motion":
            self.leds.fadeRGB("FaceLeds", self.COLOR_MOTION, 0.1)
            self.current_color = "motion"

    def set_sound_detected(self):
        """
        Set eye LEDs to green, indicating active sound detection.
        No-op if the LEDs are already showing the sound colour.
        """
        if self.current_color != "sound":
            self.leds.fadeRGB("FaceLeds", self.COLOR_SOUND, 0.1)
            self.current_color = "sound"

    def set_idle(self):
        """
        Set eye LEDs to dim white, indicating idle/no active detection.
        No-op if the LEDs are already in the idle state.
        """
        if self.current_color != "idle":
            self.leds.fadeRGB("FaceLeds", self.COLOR_IDLE, 0.1)
            self.current_color = "idle"

    def set_no_detection(self):
        """
        Set eye LEDs to dim white with a slower fade, indicating a watching
        (no current detection) state. No-op if already in this state.
        """
        if self.current_color != "no_detection":
            self.leds.fadeRGB("FaceLeds", self.COLOR_IDLE, 0.3)
            self.current_color = "no_detection"

    def reset(self):
        """
        Restore the FaceLeds group to their hardware default state and clear
        the cached colour so the next call always triggers a real LED update.
        """
        try:
            self.leds.reset("FaceLeds")
            self.current_color = None
        except Exception:
            pass


# Attention Strategies
class AttentionBase(object):
    """
    Interface:
      - start()
      - stop()
      - step(t_now) -> dict with:
          attention_source (string),
          target (dict or None): {yaw, pitch} (radians),
          bbox_ang (dict or None),
          confidence (float or None),
          extra (dict)
    """
    name = "base"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller=None):
        """
        Store shared NAO service proxies and camera parameters used by all strategies.

        Args:
            mem: ALMemory proxy for reading detection events.
            motion: ALMotion proxy for reading and setting head angles.
            video: ALVideoDevice proxy for camera access.
            camera_id (int): NAO camera index (0 = top, 1 = bottom).
            hfov (float): Camera horizontal field of view in radians.
            vfov (float): Camera vertical field of view in radians.
            led_controller (LEDController or None): Optional LED controller;
                pass None to suppress LED updates from this strategy.
        """
        self.mem = mem
        self.motion = motion
        self.video = video
        self.camera_id = camera_id
        self.hfov = hfov
        self.vfov = vfov
        self.led_controller = led_controller

    def start(self):
        """
        Subscribe to any required NAO services and begin detection.
        Override in subclasses to activate the relevant ALModule subscription.
        """
        pass

    def stop(self):
        """
        Unsubscribe from NAO services and halt detection cleanly.
        Override in subclasses to release the relevant ALModule subscription.
        """
        pass

    def step(self, t_now):
        """
        Query the current detection state and return an attention result dict.
        Override in subclasses with real detection logic.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result with keys:
                - attention_source (str): Source identifier (e.g. 'face', 'motion').
                - target (dict or None): {'yaw': float, 'pitch': float} in radians,
                  or None if no detection.
                - bbox_ang (dict or None): Angular bounding box descriptor, or None.
                - confidence (float or None): Detection confidence, or None.
                - extra (dict): Additional metadata for logging.
        """
        return {
            "attention_source": "idle",
            "target": None,
            "bbox_ang": None,
            "confidence": None,
            "extra": {}
        }

class FaceAttention(AttentionBase):
    name = "face"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov, session,
                 led_controller=None, recent_window=2.0):
        """
        Initialise face attention with the ALFaceDetection service.

        Args:
            mem: ALMemory proxy.
            motion: ALMotion proxy for reading head angles and joint limits.
            video: ALVideoDevice proxy (passed through to base class).
            camera_id (int): NAO camera index.
            hfov (float): Camera horizontal field of view in radians.
            vfov (float): Camera vertical field of view in radians.
            session: Active qi.Session used to obtain ALFaceDetection proxy.
            led_controller (LEDController or None): Optional LED controller.
            recent_window (float): Maximum in seconds for a detection event
                to be considered current (default 2.0).
        """
        AttentionBase.__init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller)
        self.recent_window = recent_window
        self.face_det = session.service("ALFaceDetection")
        self.sub_name = "attn_face_capture"
        self.motion = motion

    def start(self):
        """
        Subscribe to ALFaceDetection and begin publishing to ALMemory.
        """
        self.face_det.subscribe(self.sub_name)
        print("[FACE] Face detection started - watching for faces...")

    def stop(self):
        """
        Unsubscribe from ALFaceDetection, suppressing further face events.
        """
        try:
            self.face_det.unsubscribe(self.sub_name)
        except Exception:
            pass

    def step(self, t_now):
        """
        Read the latest FaceDetected event and compute a head movement target.
        Skips stale events older than recent_window seconds.
        Picks the first valid face in the event payload, computes an absolute
        head target from the current head angles plus the face's angular offset,
        and updates the LEDs.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result (see AttentionBase.step for key descriptions),
            extra keys: azimuth, elevation, label, scoreReco, num_faces.
        """
        val, ts = safe_get_timestamp(self.mem, "FaceDetected")
        if ts is None or (t_now - ts) > self.recent_window or not val:
            if self.led_controller:
                self.led_controller.set_no_detection()
            return {
                "attention_source": "face",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"status": "no_recent_face"}
            }

        # Filter out invalid faces
        valid_faces = []
        for face_info in val[1]:
            if isinstance(face_info, list) and len(face_info) > 0:
                shape_info = face_info[0]
                if isinstance(shape_info, list) and len(shape_info) == 5:
                    valid_faces.append(face_info)

        faces_block = valid_faces

        try:
            face_info = faces_block[0]
            shape_info = face_info[0]
            extra_info = face_info[1] if len(face_info) > 1 else None

            alpha = float(shape_info[1])
            beta = float(shape_info[2])
            size_x = float(shape_info[3])
            size_y = float(shape_info[4])

            score_reco = None
            label = None
            if extra_info and len(extra_info) >= 3:
                try:
                    score_reco = float(extra_info[1])
                except Exception:
                    score_reco = None
                label = extra_info[2]

            bbox_ang = {"mode": "center_size", "alpha": alpha, "beta": beta,
                        "sizeX": size_x, "sizeY": size_y}

            cur_yaw = float(self.motion.getAngles("HeadYaw", True)[0])
            cur_pitch = float(self.motion.getAngles("HeadPitch", True)[0])

            # Convert relative ROI offset -> absolute joint targets
            try:
                yaw_lim = self.motion.getLimits("HeadYaw")[0]
                pit_lim = self.motion.getLimits("HeadPitch")[0]
            except Exception:
                print("[FACE] Warning: cannot get head yaw and pitch limits!")
                yaw_lim = [-2,2]
                pit_lim = [-0.6,0.5]

            target = {
                "yaw": clamp(cur_yaw + alpha, yaw_lim[0], yaw_lim[1]),
                "pitch": clamp(cur_pitch + beta, pit_lim[0], pit_lim[1])
            }

            num_faces = len(faces_block)
            label_str = " (label: {})".format(label) if label else ""
            conf_str = " conf={:.2f}".format(score_reco) if score_reco is not None else ""
            print("[FACE] >>> FACE DETECTED! {} face(s){}{} | yaw={:.2f} pitch={:.2f}".format(
                num_faces, label_str, conf_str, alpha, beta))

            if self.led_controller:
                self.led_controller.set_face_detected()

            return {
                "attention_source": "face",
                "target": target,
                "bbox_ang": bbox_ang,
                "confidence": score_reco if score_reco is not None else 1.0,
                "extra": {"azimuth": alpha, "elevation": beta, "label": label, "scoreReco": score_reco, "num_faces": num_faces}
            }
        except Exception as e:
            print("[FACE] Error parsing face data: {}".format(e))
            return {
                "attention_source": "face",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"error": str(e)}
            }

class MotionAttention(AttentionBase):
    name = "motion"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov, session,
                 led_controller=None, recent_window=2.0):
        """
        Initialise motion attention with the ALMovementDetection service.

        Args:
            mem: ALMemory proxy.
            motion: ALMotion proxy for reading head angles and joint limits.
            video: ALVideoDevice proxy (passed through to base class).
            camera_id (int): NAO camera index.
            hfov (float): Camera horizontal field of view in radians.
            vfov (float): Camera vertical field of view in radians.
            session: Active qi.Session used to obtain ALMovementDetection proxy.
            led_controller (LEDController or None): Optional LED controller.
            recent_window (float): Maximum in seconds for a detection event
                to be considered current (default 2.0).
        """
        AttentionBase.__init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller)
        self.recent_window = recent_window
        self.move_det = session.service("ALMovementDetection")
        self.sub_name = "attn_motion_capture"

    def start(self):
        """
        Subscribe to ALMovementDetection and begin publishing motion clusters.
        """
        self.move_det.subscribe(self.sub_name)
        print("[MOTION] Movement detection started - watching for motion...")

    def stop(self):
        """ 
        Unsubscribe from ALMovementDetection, halting motion event publishing
        """
        try:
            self.move_det.unsubscribe(self.sub_name)
        except Exception:
            pass

    def step(self, t_now):
        """
        Read the latest MovementDetection event and compute a head movement target.
        Selects the cluster with the highest propMoving score as the attention target.
        Skips stale events older than recent_window seconds or payloads with no clusters.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result (see AttentionBase.step for key descriptions),
                extra keys: azimuth, elevation, movement_ts, posCog,
                propMoving, propAux, cur_head, num_clusters.
        """
        val, ts = safe_get_timestamp(self.mem, "MovementDetection/MovementInfo")
        if ts is None or (t_now - ts) > self.recent_window or not val:
            if self.led_controller:
                self.led_controller.set_no_detection()
            return {
                "attention_source": "motion",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"status": "no_recent_motion"}
            }

        try:
            clusters = val[1]
            if not clusters:
                if self.led_controller:
                    self.led_controller.set_no_detection()
                return {
                    "attention_source": "motion",
                    "target": None,
                    "bbox_ang": None,
                    "confidence": None,
                    "extra": {"status": "empty_clusters", "movement_ts": ts}
                }

            cur_yaw = float(self.motion.getAngles("HeadYaw", True)[0])
            cur_pitch = float(self.motion.getAngles("HeadPitch", True)[0])

            best = None
            best_prop = -1.0
            for c in clusters:
                prop_list = c[2]
                if isinstance(prop_list, (list, tuple)) and len(prop_list) > 0:
                    prop = float(prop_list[1])
                else:
                    prop = float(prop_list)
                if prop > best_prop:
                    best_prop = prop
                    best = c

            pos_cog = best[0]
            ang_roi = best[1]
            prop_list = best[2]

            prop = float(prop_list[0]) if isinstance(prop_list, (list, tuple)) and len(prop_list) > 0 else float(
                prop_list)
            prop_aux = float(prop_list[1]) if isinstance(prop_list, (list, tuple)) and len(prop_list) > 1 else None

            bbox_ang = {
                "mode": "center_size",
                "alpha": float(pos_cog[0]), "beta": float(pos_cog[1]),
                "sizeX": float(ang_roi[2]), "sizeY": float(ang_roi[3])
            }

            yaw_lim = self.motion.getLimits("HeadYaw")[0]
            pit_lim = self.motion.getLimits("HeadPitch")[0]
            dt_yaw =  pos_cog[0]
            dt_pitch = pos_cog[1]
            target = {
                "yaw": clamp(cur_yaw + dt_yaw, yaw_lim[0], yaw_lim[1]),
                "pitch": clamp(cur_pitch + dt_pitch, pit_lim[0], pit_lim[1])
            }

            print(
                "[MOTION] >>> MOTION DETECTED! {} cluster(s) | propMoving={:.1%} | target: yaw={:.2f} pitch={:.2f}".format(
                    len(clusters), prop, target["yaw"], target["pitch"]))

            if self.led_controller:
                self.led_controller.set_motion_detected()

            return {
                "attention_source": "motion",
                "target": target,
                "bbox_ang": bbox_ang,
                "confidence": prop,
                "extra": {
                    "azimuth": dt_yaw,
                    "elevation": dt_pitch,
                    "movement_ts": ts,
                    "posCog": pos_cog,
                    "propMoving": prop,
                    "propAux": prop_aux,
                    "cur_head": {"yaw": cur_yaw, "pitch": cur_pitch},
                    "num_clusters": len(clusters)
                }
            }

        except Exception as e:
            print("[MOTION] Error parsing motion data: {}".format(e))
            return {
                "attention_source": "motion",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"error": str(e), "movement_ts": ts}
            }

class SoundAttention(AttentionBase):
    name = "sound"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov, session,
                 led_controller=None, recent_window=2.0): # recent window should not be longer than image interval
        """
        Initialise sound attention with the ALSoundLocalization service.

        Args:
            mem: ALMemory proxy.
            motion: ALMotion proxy for reading head angles and joint limits.
            video: ALVideoDevice proxy (passed through to base class).
            camera_id (int): NAO camera index.
            hfov (float): Camera horizontal field of view in radians.
            vfov (float): Camera vertical field of view in radians.
            session: Active qi.Session used to obtain ALSoundLocalization proxy.
            led_controller (LEDController or None): Optional LED controller.
            recent_window (float): Maximum in seconds for a sound event to be
                considered current; should not exceed the capture interval (default 2.0).
        """
        AttentionBase.__init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller)
        self.recent_window = recent_window
        self.sound_loc = session.service("ALSoundLocalization")
        self.sub_name = "attn_sound_capture"
        self.motion = motion

    def start(self):
        """
        Configure ALSoundLocalization parameters and subscribe to begin localization.
        Sets Sensitivity to 1.0 and disables EnergyComputation for lower latency.
        """
        try:
            self.sound_loc.setParameter("EnergyComputation", False)
            self.sound_loc.setParameter("Sensitivity", 1.0) # default is 0.9
        except Exception:
            pass
        self.sound_loc.subscribe(self.sub_name)
        print("[SOUND] Sound localization started - listening for sounds...")

    def stop(self):
        """
        Unsubscribe from ALSoundLocalization, halting sound event publishing.
        """
        try:
            self.sound_loc.unsubscribe(self.sub_name)
        except Exception:
            pass

    def step(self, t_now):
        """
        Read the latest SoundLocated event and compute an absolute head target.
        Discards events older than recent_window seconds or with confidence below 0.5.
        Converts the sound's azimuth/elevation (relative to the current head pose)
        to absolute joint angles and clamps them to the NAO joint limits.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result (see AttentionBase.step for key descriptions),
                extra keys: azimuth, elevation, energy.
        """
        val, ts = safe_get_timestamp(self.mem, "ALSoundLocalization/SoundLocated")
        try:
            dt_yaw = float(val[1][0])
            dt_pitch = float(val[1][1])
            conf = float(val[1][2])
            energy = float(val[1][3])
            current_yaw = val[2][5]
            current_pitch = val[2][4]
        except Exception as e:
            print("[SOUND] Error parsing sound data: {}".format(e))
            return {
                "attention_source": "sound",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "energy": None,
                "extra": {"error": str(e)}
            }

        # only sounds after last check and with more than 50% confidence should be attended
        if ts is None or (t_now - ts) > self.recent_window or not val or conf < 0.5:
            if self.led_controller:
                self.led_controller.set_no_detection()
            return {
                "attention_source": "sound",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"status": "no_recent_sound"}
            }
        else:
            target_yaw = current_yaw + dt_yaw
            target_pitch = current_pitch+ dt_pitch
            target_yaw_norm = normalize_angle(target_yaw)
            target_pitch_norm = normalize_angle(target_pitch)



            # Convert relative ROI offset -> absolute joint targets
            try:
                yaw_lim = self.motion.getLimits("HeadYaw")[0]
                pit_lim = self.motion.getLimits("HeadPitch")[0]
            except Exception:
                print("[SOUND] Warning: cannot get head yaw and pitch limits!")
                yaw_lim = [-2, 2]
                pit_lim = [-0.6, 0.5]

            # clamp to NAO joint ranges

            target_yaw = clamp(target_yaw_norm, yaw_lim[0], yaw_lim[1])
            target_pitch = clamp(target_pitch_norm, pit_lim[0], pit_lim[1])

            target = {
                "yaw": target_yaw,
                "pitch": target_pitch
            }

            energy_str = " energy={:.2f}".format(energy) if energy is not None else ""
            print(
                "[SOUND] >>> SOUND DETECTED! azimuth={:.2f} elevation={:.2f} conf={:.2f}{} | target: yaw={:.2f} pitch={:.2f}".format(
                    dt_yaw, dt_pitch, conf, energy_str, target_yaw, target_pitch))

            if self.led_controller:
                self.led_controller.set_sound_detected()

            return {
                "attention_source": "sound",
                "target": target,
                "bbox_ang": None,
                "confidence": conf,
                "extra": {"azimuth": dt_yaw, "elevation": dt_pitch, "energy": energy}
            }

class IdleRandomAttention(AttentionBase):
    name = "idle"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov,
                 led_controller=None,
                 yaw_range=(-2.0, 2.0),
                 pitch_range=(-0.6, 0.4),
                 move_interval_sec=3.0):
        AttentionBase.__init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller)
        self.yaw_range = yaw_range
        self.pitch_range = pitch_range
        self.move_interval = move_interval_sec  # only move head every N seconds
        self._next_move_time = 0.0
        self.move_count = 0

    def start(self):
        """
        Print startup message and set LEDs to idle colour.
        """
        print("[IDLE] Idle random attention started - random head movements every {:.0f}s...".format(
            self.move_interval))
        if self.led_controller:
            self.led_controller.set_idle()

    def step(self, t_now):
        """
        Generate a new random head target once per move_interval_sec; return
        target=None between moves so the head stays still.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result (see AttentionBase.step for key descriptions).
                extra keys: mode ('waiting' or 'random_search'), move_count.
        """  
        if t_now < self._next_move_time:
            return {
                "attention_source": "idle",
                "target": None,
                "bbox_ang": None,
                "confidence": 1.0,
                "extra": {"mode": "waiting", "move_count": self.move_count}
            }

        self._next_move_time = t_now + self.move_interval
        target = {
            "yaw": random.uniform(self.yaw_range[0], self.yaw_range[1]),
            "pitch": random.uniform(self.pitch_range[0], self.pitch_range[1])
        }
        self.move_count += 1
        print("[IDLE] Random move #{}: yaw={:.2f} pitch={:.2f}".format(
            self.move_count, target["yaw"], target["pitch"]))

        if self.led_controller:
            self.led_controller.set_idle()

        return {
            "attention_source": "idle",
            "target": target,
            "bbox_ang": None,
            "confidence": 1.0,
            "extra": {"mode": "random_search", "move_count": self.move_count}
        }

class CombinedAttention(AttentionBase):
    """
    Runs face, motion and sound detection every step.

    Logging  : ALL sources that have a detection are logged as separate CSV rows
               (the main loop calls det_logger.log once per source).
    Head move: Face > Motion > Sound priority.
               Only the highest-priority active source drives the head.
    LEDs     : Reflect the source that is driving the head.

    Returns a special dict with key 'all_results' (list of per-source dicts)
    plus the usual keys filled from the winning source.
    """
    name = "all"

    def __init__(self, mem, motion, video, camera_id, hfov, vfov, session, 
                 led_controller=None, Priority=["face", "motion", "sound"]):
        """
        Initialise combined attention by constructing one sub-strategy per source.
        LED updates for sub-strategies are suppressed (led_controller=None) so that
        the CombinedAttention instance controls LEDs centrally based on the winner.

        Args:
            mem: ALMemory proxy shared with all sub-strategies.
            motion: ALMotion proxy shared with all sub-strategies.
            video: ALVideoDevice proxy shared with all sub-strategies.
            camera_id (int): NAO camera index.
            hfov (float): Camera horizontal field of view in radians.
            vfov (float): Camera vertical field of view in radians.
            session: Active qi.Session passed to each sub-strategy.
            led_controller (LEDController or None): Central LED controller.
            Priority (list): Ordered list of source names defining head-movement
                priority, highest first (default ['face', 'motion', 'sound']).
        """
        self.PRIORITY = Priority
        AttentionBase.__init__(self, mem, motion, video, camera_id, hfov, vfov, led_controller)
        self.sub_strategies = [
            FaceAttention(mem, motion, video, camera_id, hfov, vfov, session,
                          led_controller=None),  # LEDs controlled centrally
            MotionAttention(mem, motion, video, camera_id, hfov, vfov, session,
                            led_controller=None),
            SoundAttention(mem, motion, video, camera_id, hfov, vfov, session,
                           led_controller=None),
        ]

    def start(self):
        """
        Start all sub-strategies and print priority order to stdout.
        """
        for s in self.sub_strategies:
            s.start()
        print("[ALL] Combined attention started (face + motion + sound)")
        print("[ALL] Priority for head movement: face > motion > sound")
        if self.led_controller:
            self.led_controller.set_idle()

    def stop(self):
        """
        Stop all sub-strategies cleanly.
        """
        for s in self.sub_strategies:
            s.stop()

    def step(self, t_now):
        """
        Run all sub-strategies, select the highest-priority detection as the head
        movement winner, update LEDs, and return a unified result dict.
        All per-source results are included under the 'all_results' key so the
        main loop can log each source independently.

        Args:
            t_now (float): Current wall-clock time in seconds.

        Returns:
            dict: Attention result (see AttentionBase.step for key descriptions) plus:
                - all_results (dict): Mapping of source name -> per-source result dict
                  for face, motion, and sound.
        """
        results = {}
        for s in self.sub_strategies:
            r = s.step(t_now)
            results[s.name] = r

        # Pick the highest-priority source that has a detection target
        winner = None
        for src in self.PRIORITY:
            if results[src].get("target") is not None:
                winner = src
                break

        # Update LEDs based on winner (or idle if nothing detected)
        if self.led_controller:
            if winner == "face":
                self.led_controller.set_face_detected()
            elif winner == "motion":
                self.led_controller.set_motion_detected()
            elif winner == "sound":
                self.led_controller.set_sound_detected()
            else:
                self.led_controller.set_no_detection()

        # Build the return dict from the winning source (or a blank if none)
        if winner:
            w = results[winner]
            return {
                "attention_source": winner,
                "target": w["target"],
                "bbox_ang": w.get("bbox_ang"),
                "confidence": w.get("confidence"),
                "extra": w.get("extra", {}),
                "all_results": results,  # carries all three for logging
            }
        else:
            return {
                "attention_source": "none",
                "target": None,
                "bbox_ang": None,
                "confidence": None,
                "extra": {"status": "no_detection"},
                "all_results": results,
            }

def build_strategy(mode, mem, motion, video, camera_id, hfov, vfov, session, led_controller):
    """
    Instantiate and return the attention strategy matching the requested mode.

    Args:
        mode (str): One of 'face', 'motion', 'sound', 'idle', 'all', 'all_nosound'.
        mem: ALMemory proxy.
        motion: ALMotion proxy.
        video: ALVideoDevice proxy.
        camera_id (int): NAO camera index.
        hfov (float): Camera horizontal field of view in radians.
        vfov (float): Camera vertical field of view in radians.
        session: Active qi.Session for obtaining additional service proxies.
        led_controller (LEDController): LED controller passed to the strategy.

    Returns:
        AttentionBase: Concrete strategy instance for the requested mode.

    Raises:
        ValueError: If mode does not match any known strategy name.
    """
    mode = (mode or "").strip().lower()
    if mode == "face":
        return FaceAttention(mem, motion, video, camera_id, hfov, vfov, session, led_controller)
    if mode == "motion":
        return MotionAttention(mem, motion, video, camera_id, hfov, vfov, session, led_controller)
    if mode == "sound":
        return SoundAttention(mem, motion, video, camera_id, hfov, vfov, session, led_controller)
    if mode == "idle":
        return IdleRandomAttention(mem, motion, video, camera_id, hfov, vfov, led_controller)
    if mode == "all_nosound":
        return CombinedAttention(mem, motion, video, camera_id, hfov, vfov, session, led_controller, Priority = ["face", "motion"])
    if mode == "all":
        return CombinedAttention(mem, motion, video, camera_id, hfov, vfov, session, led_controller, Priority = ["face", "motion", "sound"])
    raise ValueError("Unknown mode: %s (use face|motion|sound|idle|all)" % mode)


def main(ip, port,
         mode,
         session_name,
         capture_interval_sec=2.0,
         session_duration_sec=5 * 60,
         camera_id=0,
         resolution=1,  # kVGA = 1  ->  640x480
         color_space=11,  # kRGBColorSpace = 11
         fps=5.08,
         head_speed=0.25,
         video_codec="XVID",
         annotation=True):

    # ---- Output paths --------------------------------------------------------
    out_root = os.path.join(get_script_data_root(), session_name)
    os.makedirs(out_root, exist_ok=True)
    img_dir = os.path.join(out_root, "images")
    os.makedirs(img_dir, exist_ok=True)

    video_path = os.path.join(out_root, "video.avi")
    log_path = os.path.join(out_root, "detection_history.csv")

    # ---- Connect to NAO ------------------------------------------------------
    qi_url = "tcp://{}:{}".format(ip, port)

    print("=" * 60)
    print("NAO Attention Capture - Mode: {}".format(mode.upper()))
    print("=" * 60)
    print("Connecting to NAO at {}...".format(qi_url))

    session_qi = qi.Session()
    session_qi.connect(qi_url)
    print("Connected successfully!")

    mem = session_qi.service("ALMemory")
    motion = session_qi.service("ALMotion")
    video = session_qi.service("ALVideoDevice")
    posture = session_qi.service('ALRobotPosture')


    led_controller = LEDController(session_qi)
    print("LED controller initialized")

    life = session_qi.service("ALAutonomousLife")
    life.setState("disabled")
    print("Autonomous life disabled")

    # Center head
    motion.setStiffnesses("Head", 1.0)
    #turn on for crawling
    #motion.setStiffnesses("Body", 1.0)

    # comment for crawling or sitting

    #motion.wakeUp()
    posture.goToPosture("StandInit", 0.5)

    motion.angleInterpolation(
        ["HeadYaw", "HeadPitch"],
        [0.0, 0.0],
        [1.0, 1.0],
        True
    )
    print("Head centered")
    time.sleep(0.2)

    # ---- Camera subscribe ----------------------------------------------------
    client_name = "attn_capture_cam_%d" % int(now_time() * 1000)
    name_id = video.subscribeCamera(client_name, camera_id, resolution, color_space, fps)
    print("Camera subscribed (id={})".format(camera_id))

    hfov = float(video.getHorizontalFOV(camera_id))
    vfov = float(video.getVerticalFOV(camera_id))
    print("Camera FOV: hfov={:.2f} vfov={:.2f}".format(hfov, vfov))

    # ---- Grab a test frame to get image dimensions ---------------------------
    test_img = video.getImageRemote(name_id)
    img_w = int(test_img[0])
    img_h = int(test_img[1])
    print("Image size: {}x{}".format(img_w, img_h))


    # ---- OpenCV VideoWriter --------------------------------------------------
    temp_video_path = os.path.join(out_root, "video_temp.avi")

    fourcc = cv2.VideoWriter_fourcc(*video_codec)
    writer = cv2.VideoWriter(temp_video_path, fourcc, float(fps), (img_w, img_h))
    if not writer.isOpened():
        raise RuntimeError(
            "Could not open VideoWriter for '{}'. "
            "Try a different codec (e.g. MJPG).".format(temp_video_path))
    print("VideoWriter opened (TEMP): {}  (codec={}, {}fps)".format(
        temp_video_path, video_codec, fps))


    # ---- Attention strategy --------------------------------------------------
    strategy = build_strategy(mode, mem, motion, video, camera_id, hfov, vfov,
                              session_qi, led_controller)
    strategy.start()

    # ---- Detection logger ----------------------------------------------------
    t_start = now_time()
    det_logger = DetectionLogger(log_path, session_name, mode, t_start,
                                 min_log_interval_sec=0.5)

    print("-" * 60)
    print("Session: {} | Duration: {}s".format(session_name, session_duration_sec))
    print("Video  : {}".format(video_path))
    print("Log    : {}".format(log_path))

    print("-" * 60)
    print("Starting capture loop... (Ctrl+C to stop)")
    print("")

    frame_idx = 0

    # Frame timing control to maintain consistent FPS
    target_frame_duration = 1.0 / fps  # Time each frame should take
    next_frame_time = t_start
    next_capture = t_start

    try:
        while now_time() - t_start < session_duration_sec:
            # Wait until it's time for the next frame
            t = now_time()
            if t < next_frame_time:
                sleep_time = next_frame_time - t
                time.sleep(max(0.001, sleep_time))  # Sleep at least 1ms
                t = now_time()
                if t < next_capture:
                    time.sleep(0.01)
                    continue
                next_capture = t + capture_interval_sec

            try:
                yaw_lim = motion.getLimits("HeadYaw")[0]
                pit_lim = motion.getLimits("HeadPitch")[0]
            except Exception:
                print("[SOUND] Warning: cannot get head yaw and pitch limits!")
                yaw_lim = [-2, 2]
                pit_lim = [-0.6, 0.5]

            # Update next frame time
            next_frame_time = now_time() + target_frame_duration

            # -- Attention step --
            info = strategy.step(t)
            attention_source = info.get("attention_source", mode)
            target = info.get("target", None)
            bbox_ang = info.get("bbox_ang", None)
            extra = info.get("extra", {}) or {}
            all_results = info.get("all_results", None)  # only set in 'all' mode

            # -- Move head toward winning target --
            if target is not None and yaw_lim[0] < float(target["yaw"]) < yaw_lim[1]:
                try:
                    motion.setAngles(["HeadYaw", "HeadPitch"],
                                     [float(target["yaw"]), float(target["pitch"])],
                                     head_speed)
                    print('Head is moving')
                except Exception as e:
                    extra["head_move_error"] = str(e)

            # -- Capture frame from NAO --
            nao_img = video.getImageRemote(name_id)
            raw = nao_img[6]  # RGB bytes

            # Save image
            im = Image.frombytes("RGB", (img_w, img_h), bytes(raw))
            fname = "frame_%06d.png" % frame_idx
            fpath = os.path.join(img_dir, fname)
            im.save(fpath, "PNG")

            # -- Convert RGB -> BGR for OpenCV --
            rgb_array = np.frombuffer(bytes(raw), dtype=np.uint8).reshape((img_h, img_w, 3))
            bgr_frame = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

            # -- Bounding box colours per source --
            BBOX_COLORS = {
                "face": (0, 0, 255),  # red
                "motion": (255, 0, 0),  # blue
                "sound": (0, 255, 0),  # green
            }

            elapsed = t - t_start
            head_yaw = float(motion.getAngles("HeadYaw", True)[0])
            head_pitch = float(motion.getAngles("HeadPitch", True)[0])

            # ---- 'all' mode: log every source and draw every bbox ----------
            if all_results is not None:
                any_detected = False

                for src, r in all_results.items():
                    src_detected = r.get("target") is not None
                    src_conf = r.get("confidence")
                    src_bbox_ang = r.get("bbox_ang")
                    src_extra = dict(r.get("extra", {}) or {})
                    src_extra["head_driver"] = attention_source  # which source moved the head

                    # Convert and draw bbox
                    src_bbox_px = None
                    if src_bbox_ang is not None:
                        try:
                            if src_bbox_ang.get("mode") == "center_size":
                                src_bbox_px = ang_center_size_to_px_bbox(
                                    src_bbox_ang["alpha"], src_bbox_ang["beta"],
                                    src_bbox_ang["sizeX"], src_bbox_ang["sizeY"],
                                    img_w, img_h, hfov, vfov
                                )
                                x, y, w, h = src_bbox_px
                                color = BBOX_COLORS.get(src, (255, 255, 255))
                                if annotation:
                                    cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 2)
                                    # Label the box with source name
                                    cv2.putText(bgr_frame, src, (x, max(y - 4, 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                        except Exception as e:
                            src_extra["bbox_error"] = str(e)


                    if any_detected or src_detected:
                        any_detected = True

                    # Log this source (no event image path here, added separately below)
                    det_logger.log(
                        t_now=t,
                        frame_idx=frame_idx,
                        source=src,
                        detected=src_detected,
                        confidence=src_conf,
                        bbox_px=src_bbox_px,
                        head_yaw=head_yaw,
                        head_pitch=head_pitch,
                        extra=src_extra,
                    )


                # Video overlay — show all active sources
                active_srcs = [s for s, r in all_results.items() if r.get("target") is not None]
                det_text = "Attention: All | {} |  {}".format(
                    attention_source.upper(),
                    "+".join(active_srcs).upper() if active_srcs else "watching..."
                )
                detected = any_detected

            # ---- single-source mode --------------------
            else:
                if bbox_ang is not None:
                    try:
                        if bbox_ang.get("mode") == "center_size":
                            bbox_px = ang_center_size_to_px_bbox(
                                bbox_ang["alpha"], bbox_ang["beta"],
                                bbox_ang["sizeX"], bbox_ang["sizeY"],
                                img_w, img_h, hfov, vfov
                            )
                            x, y, w, h = bbox_px
                            color = BBOX_COLORS.get(attention_source, (0, 0, 255))
                            cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), color, 2)
                    except Exception as e:
                        extra["bbox_error"] = str(e)

                detected = (target is not None)
                det_text = "Attention: {} | {}".format(
                    attention_source.upper(),
                    "DETECTED" if detected else "watching..."
                )

            # -- Video overlay  --
            cv2.putText(bgr_frame, "{:.1f}s".format(elapsed),
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if annotation:
                cv2.putText(bgr_frame, det_text,
                            (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0) if detected else (180, 180, 180), 1, cv2.LINE_AA)

            # -- Write frame to video --
            writer.write(bgr_frame)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (Ctrl+C)")
    finally:
        t_end = now_time()

        # Write footer to log
        det_logger.write_footer(t_end, frame_idx)

        # Clean shutdown
        print("\n" + "=" * 60)
        print("Shutting down...")
        try:
            strategy.stop()
            print("  - Strategy stopped")
        except Exception:
            pass
        try:
            writer.release()
            print("  - Temp VideoWriter released")
        except Exception:
            pass

        # -------------------------------------------------
        # Compute real FPS
        # -------------------------------------------------
        real_duration = t_end - t_start
        real_fps = frame_idx / real_duration if real_duration > 0 else fps

        print("  - Real FPS measured: {:.3f}".format(real_fps))

        # -------------------------------------------------
        # Re-encode video using real FPS
        # -------------------------------------------------
        print("  - Re-encoding video with real FPS...")

        try:
            cap = cv2.VideoCapture(temp_video_path)
            fourcc = cv2.VideoWriter_fourcc(*video_codec)
            final_writer = cv2.VideoWriter(video_path, fourcc, real_fps, (img_w, img_h))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                final_writer.write(frame)

            cap.release()
            final_writer.release()

            # Remove temporary file
            os.remove(temp_video_path)

            print("  - Final video written -> {} ({} FPS)".format(video_path, round(real_fps, 3)))

        except Exception as e:
            print("  ! Re-encoding failed: {}".format(e))
            print("  ! Keeping temporary video at: {}".format(temp_video_path))
        try:
            video.unsubscribe(name_id)
            print("  - Camera unsubscribed")
        except Exception:
            pass
        try:
            led_controller.reset()
            print("  - LEDs reset")
        except Exception:
            pass
        try:
            motion.setStiffnesses("Head", 0.0)
            print("  - Head stiffness released")
        except Exception:
            pass

        print("-" * 60)
        print("Session complete! {} frames captured ({:.1f}s)".format(
            frame_idx, t_end - t_start))
        expected_frames = int((t_end - t_start) * real_fps)
        print("Expected frames at {}fps: {} | Actual: {} ({:.1f}%)".format(
            real_fps, expected_frames, frame_idx,
            100.0 * frame_idx / expected_frames if expected_frames > 0 else 0))
        print("Video          : {}".format(video_path))
        print("Detection log  : {}".format(log_path))
        print("=" * 60)

# CLI
def parse_args(argv):
    """
    Parse a flat list of ``--key value`` command-line tokens into a settings dict.
    Applies type coercion and validates that the required --ip and --session
    arguments are present.

    Args:
        argv (list): Argument vector.

    Returns:
        dict: Parsed settings with keys: ip, port, mode, session, duration,
              camera, codec, fps, annotation.

    Raises:
        ValueError: If an unrecognised argument flag is encountered or if
                    --ip or --session are missing.
    """
    args = {
        "ip": None,
        "port": 9559,
        "mode": "idle",
        "session": None,
        "duration": 16 * 60,
        "camera": 0,
        "codec": "XVID",
        "fps": 5.08,
        "annotation": True
    }

    arg_types = {
        "--ip": str,
        "--port": int,
        "--mode": str,
        "--session": str,
        "--duration": float,
        "--camera": int,
        "--codec": str,
        "--fps": float,
        "--annotation": lambda v: v.lower() in ["true", "1"]
    }

    i = 1
    while i < len(argv):
        a = argv[i]
        if a not in arg_types:
            raise ValueError(f"Unknown arg: {a}")

       
        args[a.lstrip("--")] = arg_types[a](argv[i + 1])
        i += 2

    # Required checks
    if not args["ip"] or not args["session"]:
        raise ValueError("Missing required args --ip and/or --session")

    return args

if __name__ == "__main__":
    try:
        args = parse_args(sys.argv)
    except Exception as e:
        print("Error: %s" % str(e))
        print("")
        print("Usage:")
        print("  python nao_attention_with_event_images.py --ip <NAO_IP> [--port 9559] \\")
        print("         --mode face|motion|sound|idle|all --session <name> \\")
        print("         [--duration 900] [--camera 0] [--fps 6] [--codec XVID]")
        print("         [--no-events] [--event-interval 2.0]")
        print("         [--face-conf 0.5] [--motion-conf 0.3] [--sound-conf 0.4]")
        print("")
        print("Modes:")
        print("  face   - track faces only")
        print("  motion - track movement only")
        print("  sound  - track sound sources only")
        print("  idle   - random head movements, no detection")
        print("  all    - face+motion+sound simultaneously")
        print("           head priority: face > motion > sound")
        print("           CSV logs a row per active source each step")
        print("           event images saved for highest-priority detection")
        print("")
        print("Event Image Options:")
        print("  --no-events          Disable event image saving")
        print("  --event-interval N   Min seconds between event images (default: 2.0)")
        print("  --face-conf N        Min confidence for face events (default: 0.5)")
        print("  --motion-conf N      Min confidence for motion events (default: 0.3)")
        print("  --sound-conf N       Min confidence for sound events (default: 0.4)")
        print("")
        print("Outputs:")
        print("  ../data/<session>/video.avi             - continuous video")
        print("  ../data/<session>/detection_history.csv - detection log")
        print("  ../data/<session>/events/               - event images")
        print("    Format: event_<frame>_<source>_conf<N>_<timestamp>.jpg")
        print("")
        print("LED Colors:")
        print("  - Red:   Face detected / driving head")
        print("  - Blue:  Motion detected / driving head")
        print("  - Green: Sound detected / driving head")
        print("  - Dim:   No detection")
        sys.exit(1)

    main(
        ip=args["ip"],
        port=args["port"],
        mode=args["mode"],
        session_name=args["session"],
        session_duration_sec=args["duration"],
        camera_id=args["camera"],
        fps=args["fps"],
        video_codec=args["codec"],
        annotation=args["annotation"]
        )