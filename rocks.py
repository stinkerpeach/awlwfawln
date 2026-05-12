import cv2
import numpy as np
import pickle
import serial
import serial.tools.list_ports

def find_pico():
    for p in serial.tools.list_ports.comports():
        if "USB Serial" in p.description or "Pico" in p.description:
            return p.device
    for p in serial.tools.list_ports.comports():
        print(f"  {p.device} — {p.description}")
    return input("Enter port (e.g. COM5): ").strip()

pico_port = find_pico()
pico_ser  = serial.Serial(pico_port, 115200, timeout=1)

Y_TO_VOLTAGE = {1: 0.0, 2: 0.5, 3: 1.0, 4: 1.5, 5: 2.0, 6: 2.5}

def send_rock_update(rock_id, gx, gy):
    voltage = Y_TO_VOLTAGE.get(gy, 0.0)
    pico_ser.write(f"{rock_id},{gx},{voltage:.3f}\n".encode())

def send_rock_silence(rock_id):
    pico_ser.write(f"{rock_id},0,0.000\n".encode())


ROI_CORNERS = np.array([
    [503, 286],
    [798, 338],
    [746, 633],
    [451, 581],
], dtype=np.float32)

GW, GH = 6, 6

ROCK_LIBRARY_FILE = "rock_library.pkl"
CALIB_FILE        = "size_calibration.pkl"

BLUR_SIZE         = 21
DARK_OFFSET       = 65
MIN_BLOB_AREA     = 300
MAX_BLOB_AREA     = 60000
MOTION_THRESH     = 50
MAX_SATURATION    = 125
MATCH_THRESHOLD   = 1.7
W_SHAPE           = 1.5
W_AREA            = 1.5
STICKY_FRAMES     = 10
BORDER            = 0.5

VANISH_FRAMES = 45 

# load

with open(ROCK_LIBRARY_FILE, "rb") as f:
    library = pickle.load(f)
print(f"Loaded {len(library)} rocks: {sorted(library.keys())}")

with open(CALIB_FILE, "rb") as f:
    calib = pickle.load(f)
rock1_live_frac = calib["rock1_roi_fraction"]   # ~0.1002
print(f"rock1 live fraction = {rock1_live_frac:.4f}")

expected_frac = {
    rid: library[rid]["relative_size"] * rock1_live_frac
    for rid in library
}
print("Expected live fractions:")
for rid in sorted(expected_frac):
    print(f"  rock{rid}: {expected_frac[rid]:.4f}")


# grid and transform

def build_transforms(corners):
    src = np.float32(corners)
    dst = np.float32([[0, GH], [GW, GH], [GW, 0], [0, 0]])
    Mg  = cv2.getPerspectiveTransform(src, dst)
    Mp  = cv2.getPerspectiveTransform(dst, src)
    return Mg, Mp

def to_garden(px, py, Mg):
    pt  = np.array([[[float(px), float(py)]]], dtype="float32")
    out = cv2.perspectiveTransform(pt, Mg)
    return float(out[0][0][0]), float(out[0][0][1])

def to_pixel(gx, gy, Mp):
    pt  = np.array([[[float(gx), float(gy)]]], dtype="float32")
    out = cv2.perspectiveTransform(pt, Mp)
    return (int(out[0][0][0]), int(out[0][0][1]))

def garden_to_grid(gx_f, gy_f):
    gx = int(np.clip(np.ceil(gx_f), 1, GW))
    gy = int(np.clip(np.ceil(gy_f), 1, GH))
    return gx, gy

def build_detection_mask(shape, Mp, border=BORDER):
    pts = np.array([
        to_pixel(-border,     GH + border, Mp),
        to_pixel(GW + border, GH + border, Mp),
        to_pixel(GW + border, -border,     Mp),
        to_pixel(-border,     -border,     Mp),
    ], dtype=np.int32)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def draw_grid(display, Mp):
    for x in range(GW + 1):
        cv2.line(display, to_pixel(x, 0, Mp), to_pixel(x, GH, Mp),
                 (60, 60, 60), 1)
    for y in range(GH + 1):
        cv2.line(display, to_pixel(0, y, Mp), to_pixel(GW, y, Mp),
                 (60, 60, 60), 1)
    for x in range(1, GW + 1):
        lp = to_pixel(x - 0.5, 0, Mp)
        cv2.putText(display, str(x), (lp[0] - 5, lp[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)
    for y in range(1, GH + 1):
        lp = to_pixel(0, y - 0.5, Mp)
        cv2.putText(display, str(y), (lp[0] - 20, lp[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 100), 1)

# startup reference

def capture_reference(cap, n=20):
    print("Capturing reference — keep bed empty...")
    acc = None
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            continue
        g   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
        acc = g if acc is None else acc + g
    ref    = (acc / n).astype("uint8")
    ref_bl = cv2.GaussianBlur(ref, (BLUR_SIZE, BLUR_SIZE), 0)
    print("Reference captured.\n")
    return ref_bl

# matching

def match_rock(contour, blob_frac):
    """
    Returns (best_rock_id, best_score) or (None, inf) if no match found.
    Score = W_SHAPE * shape_score + W_AREA * area_score
    shape_score: matchShapes — 0 is perfect
    area_score:  |1 - blob_frac / expected_frac| — 0 is perfect size match
    """
    best_id    = None
    best_score = float("inf")

    for rid, entry in library.items():
        shape_score = cv2.matchShapes(contour, entry["contour"],
                                      cv2.CONTOURS_MATCH_I2, 0)
        area_score  = abs(1.0 - blob_frac / expected_frac[rid])
        score       = W_SHAPE * shape_score + W_AREA * area_score

        if score < best_score:
            best_score = score
            best_id    = rid

    return best_id, best_score

# sticky tracker

class StickyTracker:
    """
    A rock ID must hold the same grid cell for STICKY_FRAMES frames
    before its label is shown. Prevents flicker on placement.
    """
    def __init__(self):
        self._buf      = {}
        self.confirmed = {}
        self._missing  = {}

    def update(self, seen):
        for rid in list(self._buf):
            if rid not in seen:
                self._missing[rid] = self._missing.get(rid, 0) + 1
                if self._missing[rid] > VANISH_FRAMES:
                    self._buf.pop(rid, None)
                    self.confirmed.pop(rid, None)
                    self._missing.pop(rid, None)
            else:
                self._missing.pop(rid, None)

        for rid, pos in seen.items():
            if rid not in self._buf:
                self._buf[rid] = []
            self._buf[rid].append(pos)
            if len(self._buf[rid]) > STICKY_FRAMES:
                self._buf[rid].pop(0)
            if len(self._buf[rid]) == STICKY_FRAMES:
                if len(set(self._buf[rid])) == 1:
                    self.confirmed[rid] = pos
            else:
                self.confirmed.pop(rid, None)

# MAIN LOOP

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,   1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
cap.set(cv2.CAP_PROP_AUTOFOCUS,     0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

Mg, Mp = build_transforms(ROI_CORNERS)

ret, frame = cap.read()
H, W = frame.shape[:2]

roi_mask  = np.zeros((H, W), dtype=np.uint8)
cv2.fillPoly(roi_mask, [ROI_CORNERS.astype(np.int32)], 255)
roi_area  = float(cv2.countNonZero(roi_mask))
det_mask  = build_detection_mask(frame.shape, Mp)

print(f"ROI area = {roi_area:.0f} px")

ref_bl  = capture_reference(cap)
tracker = StickyTracker()

last_printed = {}

print("R: recapture reference   Q: quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv     = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_bl = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)
    display = frame.copy()

    cv2.polylines(display, [ROI_CORNERS.astype(np.int32)], True, (0, 230, 230), 2)
    draw_grid(display, Mp)

    # blob detection
    diff     = cv2.absdiff(gray_bl, ref_bl)
    dark_bin = np.zeros((H, W), dtype=np.uint8)
    dark_bin[gray_bl.astype("int16") < ref_bl.astype("int16") - DARK_OFFSET] = 255
    blob_bin = cv2.bitwise_and(dark_bin, det_mask)

    k        = np.ones((7, 7), np.uint8)
    blob_bin = cv2.morphologyEx(blob_bin, cv2.MORPH_OPEN,  k)
    blob_bin = cv2.morphologyEx(blob_bin, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(blob_bin, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    seen_this_frame = {}   # rock_id → (gx, gy)

    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA or area > MAX_BLOB_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # motion filter
        motion_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.drawContours(motion_mask, [c], -1, 255, -1)
        vals = diff[motion_mask == 255].astype("float32")
        if len(vals) == 0 or float(np.std(vals)) > MOTION_THRESH:
            cv2.drawContours(display, [c], -1, (40, 40, 40), 1)
            continue

        # saturation filter
        mean_sat = cv2.mean(hsv[:, :, 1], mask=motion_mask)[0]
        if mean_sat > MAX_SATURATION:
            cv2.drawContours(display, [c], -1, (0, 0, 140), 1)
            continue

        # match 
        blob_frac      = area / roi_area
        rock_id, score = match_rock(c, blob_frac)

        if rock_id is None or score > MATCH_THRESHOLD:
            cv2.drawContours(display, [c], -1, (0, 80, 160), 1)
            cv2.putText(display, f"? {score:.2f}",
                        (cx + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 160), 1)
            continue

        if rock_id in seen_this_frame:
            continue   # already matched this rock this frame

        gx_f, gy_f = to_garden(cx, cy, Mg)
        gx, gy     = garden_to_grid(gx_f, gy_f)

        seen_this_frame[rock_id] = (gx, gy)

        # draw candidate outline while stabilizing
        cv2.drawContours(display, [c], -1, (0, 180, 220), 1)
        cv2.putText(display, f"r{rock_id}? {score:.2f}",
                    (cx + 6, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 180, 220), 1)

    # update tracker + draw confirmed
    tracker.update(seen_this_frame)

    for rid, (gx, gy) in tracker.confirmed.items():
        px = to_pixel(gx - 0.5, gy - 0.5, Mp)
        cv2.circle(display, px, 6, (50, 220, 80), -1)
        cv2.putText(display, f"rock{rid}",
                    (px[0] - 18, px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 80), 2)
        cv2.putText(display, f"({gx},{gy})",
                    (px[0] - 18, px[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

        if last_printed.get(rid) != (gx, gy):
            send_rock_update(rid, gx, gy)
            print(f"rock{rid} | ({gx}, {gy}) | sent to Pico")
            last_printed[rid] = (gx, gy)

    for rid in list(last_printed):
        if rid not in tracker.confirmed:
            send_rock_silence(rid)
            print(f"rock{rid} removed — silenced")
            del last_printed[rid]

    cv2.putText(display,
                f"R:recapture  Q:quit  "
                f"dark={DARK_OFFSET} motion={MOTION_THRESH} "
                f"sat={MAX_SATURATION} thresh={MATCH_THRESHOLD}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    cv2.imshow("Rock Garden", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('r'), ord('R')):
        ref_bl  = capture_reference(cap)
        tracker = StickyTracker()
        last_printed.clear()

cap.release()
cv2.destroyAllWindows()