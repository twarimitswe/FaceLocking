# face-recognition

Camera face detection/recognition pipeline with an optional ESP32 pan-servo
that turns to follow one enrolled person.

All commands below are run from this directory, with the virtualenv active:

```
source .venv/bin/activate
```

## Setup (one time)

- Model files are gitignored (too large to commit) and must be downloaded
  into `models/`:
  - `models/face_landmarker.task` - MediaPipe FaceLandmarker (Tasks API)
  - `models/embedder_arcface.onnx` - ArcFace embedding model (InsightFace
    `buffalo_l/w600k_r50`, ~166MB)
- `pip install -r requirements.txt` if one exists, otherwise the packages
  used are: `opencv-contrib-python`, `mediapipe`, `numpy`, `onnxruntime`,
  `requests`.

## 1. `src/camera.py` - sanity check your camera

```
python -m src.camera
```

Auto-detects and opens a camera, preferring a plugged-in USB camera over
the laptop's built-in one (see "Camera auto-detection" below). Run this
first if nothing else is working, to confirm OpenCV can see a camera at
all. Press `q` to quit.

### Camera auto-detection

`src/camera.py`'s `pick_camera_index()` is used by every script below
that opens a camera (`track.py`, `face_tracking.py`, and this file's own
`main()`). On Linux, video devices enumerate in attach order: the
built-in webcam is present at boot and claims the lowest index(es), so a
USB camera plugged in afterward gets a higher one. Among the working,
real capture devices (checked via udev's V4L2 capability flag, since a
camera can expose a second metadata-only node that still opens and
returns a frame) it picks the highest index - the external camera if one
is plugged in, the built-in one otherwise. Pass `--camera N` on
`track.py`/`face_tracking.py` to override this and force a specific
index.

## 2. `src/detect.py` - Haar face box only

```
python -m src.detect
```

Minimal Haar cascade face detection with box smoothing (tracks the
largest face, holds the box briefly across missed frames instead of
flickering). No landmarks, no recognition - just confirms face detection
itself is working. Press `q` to quit.

## 3. `src/landmark.py` - Haar box + 5-point landmarks

```
python -m src.landmark
```

Adds MediaPipe FaceLandmarker on top of the smoothed Haar box, drawing the
5 keypoints (eyes, nose, mouth corners) used for alignment. Press `q` to
quit.

## 4. `src/align.py` - alignment preview

```
python -m src.align
```

Shows the live camera feed with box/5pt alongside the aligned 112x112 crop
(the ArcFace-ready warp). Press `s` to save the current aligned crop to
`data/debug_aligned/`, `q` to quit. Useful for checking alignment quality
before enrolling.

## 5. `src/embed.py` - embedding visualization

```
python -m src.embed
```

Runs the aligned crop through the ArcFace ONNX model and visualizes the
512-d embedding as a heatmap, plus frame-to-frame cosine similarity. Press
`p` to print embedding stats to the terminal, `q` to quit. Mostly a
debugging/educational tool - not needed for the enroll/recognize flow
below.

## 6. `src/enroll.py` - enroll a person (do this first, for real)

```
python -m src.enroll
```

Type a name when prompted, then:

- **SPACE** - capture one sample (needs a face detected)
- **a** - toggle auto-capture (captures every ~0.25s)
- **s** - save the enrollment (needs a handful of samples first)
- **r** - reset newly-captured samples this session (keeps saved crops on
  disk)
- **q** - quit

Repeat once per person. Saves to `data/db/face_db.npz` (embeddings) and
`data/enroll/<name>/*.jpg` (aligned crops, reused if you re-run enroll for
the same name later to extend their template).

## 7. `src/evaluate.py` - check recognition quality (optional)

```
python -m src.evaluate
```

Needs at least 2 enrolled people with several samples each. Prints
genuine vs. impostor cosine-distance distributions and suggests a
threshold for a target false-accept rate. Useful for sanity-checking
before relying on `recognize.py`/`track.py`.

## 8. `src/recognize.py` - live multi-face recognition (no servo)

```
python -m src.recognize
```

Detects and labels every face in frame against `data/db/face_db.npz`.

- **q** - quit
- **r** - reload the DB from disk (after enrolling someone new)
- **+ / -** - adjust the match distance threshold live
- **d** - toggle a debug overlay

## 9. `src/face_tracking.py` - identity lock, tracking, and gestures (no servo)

```
python -m src.face_tracking --target "SomeEnrolledName"
```

Locks onto one enrolled identity and tracks it with a small state machine
(`SEARCHING` -> `LOCKED` -> `LOST` -> back to `SEARCHING` after a timeout),
reusing `recognize.py`'s detector/embedder/matcher and `align.py`'s
alignment rather than duplicating them. Between full ArcFace re-checks
(every `verify_every` frames) it just associates the nearest matching box
(IoU + displacement) so it stays fast, and never transfers the lock to a
different face crossing the frame. While locked, `src/face_signals.py`
runs MediaPipe FaceLandmarker on the locked box only (never the full
frame) to report smile, blink count, and eyes-open/closed. The window also
shows normalized `error_x`/`error_y` and `LEFT/RIGHT/UP/DOWN/CENTER`
labels - the same position signal `src/track.py` computes for the servo,
here just displayed instead of driving hardware. Press `q` to quit.

Add `--debug` for verbose logging (via the stdlib `logging` module) of
every detection/tracking/signal step: per-frame face count and lock
state, each acquire/associate candidate considered with its match
similarity, every `LOCKED`/`LOST`/`SEARCHING` transition, the smoothed
position and normalized error each frame, and every blink/smile/eye-state
change with the EAR or smile score that triggered it. Without `--debug`
only state-changing events are printed (quiet by default).

Smile and blink/eyes-closed are detected relative to each person's own
resting geometry, not a fixed number shared by everyone: a running
baseline tracks your neutral mouth-width/face-width ratio and open-eye
EAR (each only drifts while that signal is NOT currently triggered, so a
sustained smile or blink can't pull its own trigger point toward itself).
Smiling fires once the ratio rises `--smile-on-margin` above your own
baseline and clears once it falls back within `--smile-off-margin` of it;
blink/eyes-closed fire once EAR drops `--ear-drop-frac` below your own
open-eye baseline. This self-calibrates per person/camera without manual
tuning; the flags exist to adjust sensitivity if needed. With `--debug`,
the `smile geometry: ... ratio=` and `signals: ear=...` lines show both
the live value and the current baseline it's being compared against.

## 10. `src/track.py` - servo face-following (needs the ESP32 hardware)

Requires `firmware/servo_tracker/servo_tracker.ino` flashed onto the ESP32
first - see the firmware section below.

```
python -m src.track --host <esp32-ip-or-servotracker.local>
```

- `--camera N`: overrides camera auto-detection (see above) with a
  specific `/dev/videoN` index; omit it to auto-prefer a plugged-in USB
  camera over the built-in one.
- `--host`: the ESP32's mDNS hostname (`servotracker.local`, if mDNS
  resolves on your machine) or its current DHCP IP (check the Arduino
  serial monitor - it's printed on boot, and can change between reboots).
- `--invert`: add this if the servo turns the wrong direction.
- `--min-angle` / `--max-angle`: restrict the safe servo sweep range if
  the mount/cabling can't handle the full 0-180 (defaults to 0-180).

Prompts you to pick which enrolled name to follow, then:

- Locks onto that person's face and holds the pan angle steady while
  they're in frame (small position noise is ignored; only a real drift
  past a deadzone triggers a correction, and after any correction it
  waits for the servo to physically finish moving plus a settle delay
  before considering another one).
- If the person is out of frame for more than ~1 second, sweeps slowly
  back and forth searching for them until reacquired.
- **c** - re-center the servo manually
- **q** - quit

## Firmware: `firmware/servo_tracker/servo_tracker.ino`

ESP32 sketch driving the pan servo (GND -> GND, signal -> D18, power ->
Vin). Before flashing, edit `WIFI_SSID` / `WIFI_PASSWORD` at the top of
the file for your **2.4GHz** network (ESP32 cannot join 5GHz-only SSIDs).

Using the Arduino IDE: install the "esp32" board package (Boards Manager
URL: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`)
and the "ESP32Servo" library, select your board/port, then upload.

Using `arduino-cli`:

```
arduino-cli compile --fqbn esp32:esp32:esp32 firmware/servo_tracker
arduino-cli upload --fqbn esp32:esp32:esp32 -p /dev/ttyUSB0 firmware/servo_tracker
```

Open the serial monitor at 115200 baud after flashing to confirm it
joined WiFi and note its IP address for `src/track.py --host`.
