import time
import numpy as np
import cv2
from ultralytics import YOLO
from distance_utils import estimate_distance_m, distance_to_proximity, distance_label

# name → priority (1=highest danger, 5=lowest)
DETECT_CLASSES = {
    # Signals
    "traffic light": 1,
    "stop sign":     1,
    # People & animals
    "person":        1,
    "dog":           3,
    "cat":           3,
    # Vehicles
    "car":           2,
    "motorcycle":    2,
    "bus":           2,
    "truck":         2,
    "bicycle":       2,
    "train":         2,
    # Architectural obstacles
    "door":          2,
    "gate":          2,
    "stairs":        2,
    "wall":          3,
    # Street furniture
    "fire hydrant":  3,
    "bench":         4,
    "chair":         3,
    "dining table":  4,
    "couch":         4,
    # For traffic light testing via phone screen
    "cell phone":    5,
}

# Traffic light color detection — ratio-based, only counts bright lit pixels
def detect_light_color(frame_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    w_b = x2 - x1
    h_b = y2 - y1

    if h_b < 20 or w_b < 8:
        return None

    # Real traffic lights are taller than wide — skip horizontal boxes
    if w_b > h_b * 1.3:
        return None

    hsv = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    t   = max(1, h_b // 3)

    def lit_ratio(y0, ye, h_lo, h_hi, h_lo2=None, h_hi2=None):
        """Ratio of bright+saturated pixels matching hue in this region."""
        region = hsv[y0:ye]
        if region.size == 0:
            return 0.0
        # Lit lamp: high brightness (V>130) + decent saturation (S>90)
        mask = cv2.inRange(region, np.array([h_lo,  90, 130]),
                                   np.array([h_hi, 255, 255]))
        if h_lo2 is not None:
            mask = cv2.bitwise_or(mask,
                   cv2.inRange(region, np.array([h_lo2,  90, 130]),
                                       np.array([h_hi2, 255, 255])))
        return cv2.countNonZero(mask) / max(1, (ye - y0) * w_b)

    red_r    = lit_ratio(0,   t,     0, 10, 165, 180)  # top third
    yellow_r = lit_ratio(t,   2 * t, 18, 35)            # middle third
    green_r  = lit_ratio(2*t, h_b,   40, 90)            # bottom third

    MIN = 0.06  # at least 6% of the section must be lit in the right colour

    if red_r    > MIN and red_r    > yellow_r   and     green_r:  return "red"
    if green_r  > MIN and green_r  > red_r      and     yellow_r: return "green"
    if yellow_r > MIN and yellow_r > red_r      and     green_r:   return "yellow"
    return None


# Solid-colour screen detection — for testing with a phone showing a plain red/green/yellow image
def detect_screen_color(frame_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv   = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    total = max(1, crop.shape[0] * crop.shape[1])

    def ratio(h_lo, h_hi, s_lo=80, v_lo=80, h_lo2=None, h_hi2=None):
        mask = cv2.inRange(hsv, np.array([h_lo,  s_lo, v_lo]),
                                np.array([h_hi, 255, 255]))
        if h_lo2 is not None:
            mask = cv2.bitwise_or(mask,
                   cv2.inRange(hsv, np.array([h_lo2, s_lo, v_lo]),
                                    np.array([h_hi2, 255, 255])))
        return cv2.countNonZero(mask) / total

    red_r    = ratio(0, 10, h_lo2=160, h_hi2=180)
    green_r  = ratio(40, 90)
    yellow_r = ratio(18, 35)

    MIN = 0.25  # at least 25% of the screen must be the dominant colour

    if red_r    > MIN and red_r    >= green_r  and red_r    >= yellow_r: return "red"
    if green_r  > MIN and green_r  > red_r     and green_r  >= yellow_r: return "green"
    if yellow_r > MIN and yellow_r > red_r     and yellow_r > green_r:   return "yellow"
    return None


def get_horizontal_zone(cx, frame_w):
    third = frame_w / 3
    if cx < third:
        return "left"
    elif cx < 2 * third:
        return "ahead"
    else:
        return "right"


def get_proximity(box_area, frame_area):
    ratio = box_area / max(frame_area, 1)
    if ratio > 0.15:
        return "close"
    elif ratio > 0.04:
        return "nearby"
    return "far"


class SceneDetector:
    def __init__(self):
        self.model = YOLO("yolov8s-worldv2.pt")
        self.model.set_classes(list(DETECT_CLASSES.keys()))
        self._last_announced: dict[str, float] = {}
        self._last_light_color: str | None = None

    def _cooldown(self, label: str, priority: int) -> float:
        # Never repeat while object is still in frame — only resets when it leaves
        return float('inf')

    def detect(self, image_bytes: bytes) -> dict:
        img_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return {"announcements": [], "detections": []}

        h, w = frame.shape[:2]
        frame_area = w * h

        # Run at lower conf so architectural classes aren't missed; filter per-class below
        results = self.model(frame, verbose=False, conf=0.18)[0]

        # Architectural classes need a lower threshold — they're harder to detect
        ARCH = {"door", "gate", "wall", "fence", "stairs", "step"}
        MIN_CONF = 0.18   # for architectural
        STD_CONF = 0.25   # for everything else

        detections = []
        candidates = []
        visible_keys: set[str] = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label    = results.names[cls_id]
            priority = DETECT_CLASSES.get(label, 4)
            conf     = float(box.conf[0])

            # Per-class confidence gate
            if conf < (MIN_CONF if label in ARCH else STD_CONF):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            box_w = x2 - x1
            box_h = y2 - y1
            box_area   = box_w * box_h
            zone       = get_horizontal_zone(cx, w)
            distance_m = estimate_distance_m(label, [x1, y1, x2, y2])
            proximity  = distance_to_proximity(distance_m)

            # Traffic lights are always relevant — skip only if truly tiny
            if label == "traffic light":
                if box_w < 15 or box_h < 15:
                    continue
            elif box_w < 60 or box_h < 60 or proximity == "far":
                continue

            extra = ""
            if label == "traffic light":
                color = detect_light_color(frame, (x1, y1, x2, y2))
                if color:
                    extra = color

            # Cell phone showing a solid red/green/yellow screen (used for testing)
            # → reclassify as traffic light anywhere in frame, skip if no matching colour
            if label == "cell phone":
                color = detect_screen_color(frame, (x1, y1, x2, y2))
                if color:
                    label, priority, extra = "traffic light", 1, color
                    # Re-check size with lenient threshold now that it's a traffic light
                    if box_w < 15 or box_h < 15:
                        continue
                else:
                    continue  # ignore phones not showing a signal color

            # Key is just label — same object won't repeat even if it moves, until it leaves frame
            key = f"traffic_light_{extra}" if label == "traffic light" and extra else label
            visible_keys.add(key)

            detections.append({
                "label":      label,
                "conf":       round(conf, 2),
                "zone":       zone,
                "proximity":  proximity,
                "distance_m": distance_m,
                "extra":      extra,
                "box":        [round(x1), round(y1), round(x2), round(y2)],
                "priority":   priority,
            })
            candidates.append({
                "label": label,
                "priority": priority,
                "zone": zone,
                "proximity": proximity,
                "extra": extra,
                "key": key,
                "zone": zone,
            })

        # Reset cooldown for objects that LEFT the frame — so re-entry announces again
        for key in list(self._last_announced.keys()):
            if key not in visible_keys:
                del self._last_announced[key]
        if self._last_light_color and f"traffic_light_{self._last_light_color}" not in visible_keys:
            self._last_light_color = None

        proximity_order = {"close": 0, "nearby": 1, "far": 2}
        candidates.sort(key=lambda d: (d["priority"], proximity_order.get(d["proximity"], 3)))

        now = time.time()
        announcements = []
        seen_keys: set[str] = set()

        for c in candidates:
            key = c["key"]
            label = c["label"]

            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Traffic light color change always announces
            if label == "traffic light" and c["extra"] and c["extra"] != self._last_light_color:
                self._last_light_color = c["extra"]
                self._last_announced[key] = 0

            cooldown = self._cooldown(label, c["priority"])
            if now - self._last_announced.get(key, 0) < cooldown:
                continue

            self._last_announced[key] = now
            phrase = self._build_phrase(c)
            if phrase:
                announcements.append(phrase)

        nav = self._navigation_advice(detections, w, h)
        return {"announcements": announcements, "detections": detections, "frame_size": [w, h], "nav": nav}

    def _navigation_advice(self, detections: list, frame_w: int, frame_h: int) -> dict:
        # Walking corridor: centre 50% of frame width, full height
        wz_x1 = frame_w * 0.25
        wz_x2 = frame_w * 0.75

        # Traffic lights are always relevant regardless of walk zone
        traffic_lights = [d for d in detections if d["label"] == "traffic light" and d["extra"]]
        tl_info = None
        if traffic_lights:
            c = traffic_lights[0]["extra"]
            tl_info = {
                "color":   c,
                "message": ("Red light. Stop. Do not cross."  if c == "red"    else
                            "Green light. Safe to walk."       if c == "green"  else
                            "Yellow light. Prepare to stop."),
            }

        # Any detection whose box overlaps the walking corridor is a blocker
        blockers = [
            d for d in detections
            if d["box"][2] > wz_x1 and d["box"][0] < wz_x2
        ]

        walk_zone = [0.25, 0.75]  # sent to frontend for drawing

        if not blockers:
            return {
                "status": "clear",
                "message": "Path ahead is clear, safe to walk",
                "walk_zone": walk_zone,
                "traffic_light": tl_info,
            }

        prox_rank = {"close": 0, "nearby": 1}
        main = sorted(blockers, key=lambda d: (d["priority"], prox_rank.get(d["proximity"], 2)))[0]

        label = main["label"]
        cx = (main["box"][0] + main["box"][2]) / 2

        # Danger score for the side corridors (outside the walk zone)
        def side_danger(sx1, sx2):
            score = 0
            for d in detections:
                bx1, _, bx2, _ = d["box"]
                if bx2 > sx1 and bx1 < sx2:
                    score += 3 if d["proximity"] == "close" else 1
            return score

        left_danger  = side_danger(0, wz_x1)
        right_danger = side_danger(wz_x2, frame_w)

        if left_danger < right_danger:
            go = "left"
        elif right_danger < left_danger:
            go = "right"
        else:
            # Go away from where the blocker sits inside the corridor
            go = "left" if cx > frame_w / 2 else "right"

        center_offset = abs(cx - frame_w / 2) / (frame_w / 2)
        base  = 45 if main["proximity"] == "close" else 25
        angle = max(15, round(base * (1 - center_offset * 0.3) / 5) * 5)

        # Architectural obstacles always get a warning regardless of proximity
        arch_labels = {"door", "gate", "wall", "fence", "stairs", "step"}
        dist_str  = distance_label(main.get("distance_m"))
        dist_part = f", {dist_str}" if dist_str else ""

        if label in arch_labels:
            if main["proximity"] == "close":
                warning = "stop, obstacle" if label in ("wall", "fence") else "please be careful"
                message = f"{label.capitalize()} ahead — {warning}. Move {go}"
            else:
                message = f"{label.capitalize()} ahead{dist_part} — please be careful. Move {go}"
        elif main["proximity"] == "close":
            message = f"Move {go}"
        else:
            message = f"{label.capitalize()} is nearby{dist_part}. Move {go}"

        return {
            "status": "blocked",
            "message": message,
            "direction": go,
            "angle": angle,
            "obstacle": label,
            "walk_zone": walk_zone,
            "traffic_light": tl_info,
        }

    def _build_phrase(self, c: dict) -> str:
        label      = c["label"]
        zone       = c["zone"]
        proximity  = c["proximity"]
        extra      = c["extra"]
        dist_str   = distance_label(c.get("distance_m"))

        zone_phrase  = f"on the {zone}" if zone != "ahead" else "ahead"
        # Nearby: include distance. Close: skip distance, add urgency
        dist_suffix  = f", {dist_str}" if dist_str and proximity == "nearby" else ""
        close_suffix = " — very close" if proximity == "close" else ""

        # Special cases
        if label == "traffic light":
            if extra == "red":    return "Red light. Stop. Do not cross."
            if extra == "green":  return "Green light. Safe to walk."
            if extra == "yellow": return "Yellow light. Prepare to stop."
            return f"Traffic light {zone_phrase}"

        if label == "stop sign":
            return "Stop sign ahead"

        if label == "person":
            return f"Person {zone_phrase}{dist_suffix}{close_suffix}"

        if label in ("car", "truck", "bus", "motorcycle", "bicycle", "train"):
            return f"{label.capitalize()} {zone_phrase}{dist_suffix}{close_suffix}"

        if label in ("dog", "cat"):
            return f"{label.capitalize()} {zone_phrase}{dist_suffix}{close_suffix}"

        if label == "fire hydrant":
            return f"Fire hydrant {zone_phrase}{dist_suffix}"

        if label == "chair":
            return f"Chair {zone_phrase}{dist_suffix}{close_suffix}"

        if label == "bench":
            return f"Bench {zone_phrase}{dist_suffix}"

        if label == "dining table":
            return f"Table {zone_phrase}{dist_suffix}"

        if label == "door":
            if proximity == "close":
                return f"Door {zone_phrase}{close_suffix} — please be careful"
            return f"Door {zone_phrase}{dist_suffix} — please be careful"

        if label == "gate":
            if proximity == "close":
                return f"Gate {zone_phrase}{close_suffix} — please be careful"
            return f"Gate {zone_phrase}{dist_suffix} — please be careful"

        if label in ("wall"):
            if proximity == "close":
                return f"{label.capitalize()} {zone_phrase} — stop, obstacle ahead"
            return f"{label.capitalize()} {zone_phrase}{dist_suffix} — please be careful"

        if label in ("stairs"):
            if proximity == "close":
                return f"{'Stairs' if label == 'stairs' else 'Step'} {zone_phrase} — watch your step carefully"
            return f"{'Stairs' if label == 'stairs' else 'Step'} {zone_phrase}{dist_suffix} — watch your step"

        # Generic fallback
        return f"{label.capitalize()} {zone_phrase}{dist_suffix}"
