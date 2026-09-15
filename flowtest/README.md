# IDDSI Flow-Test video grader (deterministic v0)

This package measures the residual volume in a filmed **10 mL / 10 second syringe Flow Test** with deterministic OpenCV. It uses no ML, network service, or GPU. It is an open research and culinary-QA measurement aid—not a swallowing-safety decision system.

The grader detects a 61.5 mm calibrated syringe barrel, detects release from a nozzle/finger change or the start of liquid movement, rectifies the barrel, tracks the meniscus, and reads it exactly 10 seconds after release. It maps residual volume as follows:

| Reading | Output |
|---|---|
| `< 1 mL` | L0 |
| `1–<4 mL` | L1 |
| `4–<8 mL` | L2 |
| `>=8 mL` after visible flow | L3 |
| No detectable flow | L4 |

Any reading within **±0.5 mL** of 1, 4, or 8 mL abstains. A numeric estimate is retained for audit, but no level is returned. The grader also abstains on an unverified/wrong syringe, excessive tilt, occlusion, bubbles/lumps, unclear meniscus, missing release, or a clip ending before the 10-second frame.

### Abstention codes

Every `abstain` result lists machine-readable codes in `abstention_reasons`, and `diagnostics.abstention_code_descriptions` maps each code to a human explanation. The canonical registry — code, description, and corrective capture action — lives in `flowtest/abstention.py` (`ABSTENTION_CODES`) and is exported as `flowtest.ABSTENTION_CODES`:

| Code | Meaning |
|---|---|
| `syringe_not_detected` | No supported barrel geometry found in the first frame |
| `scale_not_verified` | No ID-1 fiducial detected; absolute scale unverified |
| `wrong_or_unverified_syringe` | Barrel length outside 61.5 ± 3.0 mm, or unverifiable |
| `syringe_not_vertical` | Barrel axis tilt beyond the configured limit |
| `release_not_detected` | No release cue or liquid start detected |
| `insufficient_duration` | Clip ends before the 10 s post-release frame |
| `meniscus_unclear` | Meniscus contrast below the confidence limit |
| `syringe_occluded` | Barrel edges obscured beyond the allowed fraction |
| `bubbles_or_lumps_detected` | Contrasting regions inside the residual liquid |
| `near_level_boundary` | Reading within the margin of the 1/4/8 mL boundaries |
| `uncalibrated_scale_mode` | Audit flag: graded without a fiducial (see below); does not by itself force abstention |

### Fiducial-optional (uncalibrated) mode

By default a missing ID-1 card abstains with `scale_not_verified`. Passing `--fiducial-optional` to the CLI (or `FlowTestConfig(fiducial_required=False)` in code) lets the grader proceed without the card: the `scale_fiducial` and `correct_syringe` checks become informational, `diagnostics.scale_mode` reads `"uncalibrated"`, and the `uncalibrated_scale_mode` audit code is attached. Other quality checks still apply and can still abstain. Use this only when a relative, unverified-scale reading is acceptable.

### Rotation and non-ASCII paths

Phone rotation metadata (90/180/270 display rotation) is honoured: if the OpenCV build supports auto-orientation it is used, otherwise frames are rotated during decode (`diagnostics.rotation_deg` / `rotation_applied` report what happened). Non-ASCII filenames such as `增稠剂.mp4` are opened directly; if an OpenCV build rejects them, the file is copied to a temporary ASCII name (the original is never modified or deleted) and decoded from there (`diagnostics.ascii_path_fallback`).

## Capture protocol

1. Put a high-contrast, dark-bordered **85.60 × 53.98 mm ID-1 card** in the same plane as the syringe. A blank printed rectangle is sufficient; do not resize it when printing. This v0 needs the card to verify the 61.5 mm barrel rather than guessing absolute scale.
2. Use the official 10 mL syringe geometry with a 61.5 mm calibrated barrel. Mount it vertically and show the whole calibrated section plus nozzle.
3. Use a fixed side view, diffuse light, a plain contrasting background, and at least 8 px across each mL interval. Avoid glare, camera motion, hands over the barrel, foam, bubbles, and lumps.
4. Start filming before the outlet is released and continue for more than 10 seconds afterward. Fill to exactly 10 mL and follow the official test instructions, including the required liquid temperature and preparation conditions.
5. Treat `abstain` as “unclear—repeat the official physical test,” never as a pass.

The simple rectangular fiducial is intentionally dependency-free. Because it is not a uniquely coded marker, keep other card-shaped objects out of frame.

## Install and run

Python 3.10+ is required. From this directory:

```bash
python -m pip install -e '.[test]'
python -m flowtest.grade video.mp4 --out result.json
```

The JSON contains the measured residual volume, release/measurement timestamps, every quality check and its threshold, diagnostics, abstention reasons, and the research-use notice. A successfully decoded but unsuitable capture returns JSON with `status: "abstain"`; unreadable input returns `status: "error"` and exit code 2.

## Synthetic end-to-end fixtures

Generate one video:

```bash
python -m flowtest.synthetic --out synthetic.mp4 --residual-ml 6.0
python -m flowtest.grade synthetic.mp4 --out result.json
```

Generate a five-case suite and calculate measured synthetic MAE:

```bash
python -m flowtest.synthetic --evaluate synthetic_outputs
```

This writes `synthetic_mae.json`. Its MAE is explicitly labelled as a **synthetic pipeline-development check**. It is not an accuracy result and must not be presented as real-world or clinical validation.

In the development run on 31 August 2026, the default five generated fixtures (0.25, 2.5, 6.0, 9.0, and 10.0 mL residual) produced **0.0918 mL synthetic MAE**. The per-case errors were 0.239, 0.138, 0.059, 0.023, and 0.000 mL. These figures are measured from generated geometric drawings using the same code that created them; codec/build differences may change them, and they say nothing about performance on camera footage.

Run tests with:

```bash
pytest
```

The tests cover end-to-end measurement, exact band mapping, boundary abstention, wrong-syringe rejection, tilt rejection, the CLI, and synthetic MAE reporting. Fixtures are small CPU-generated MP4 files.

## Algorithm

The first frame is thresholded to find the ID-1 rectangle and establish pixels/mm. Canny/Hough line pairs locate the long barrel sides and determine its angle and 61.5 mm length. A perspective transform normalises the barrel. Each frame's meniscus candidate combines HSV liquid/air separation with the horizontal grayscale gradient. A nozzle-region change detects finger release; meniscus displacement is the fallback. The median in a ±0.1-second window around `t0 + 10 s` becomes the residual reading.

Quality checks are conservative deterministic heuristics: edge visibility estimates occlusion, connected high-contrast regions estimate bubbles/lumps, and meniscus contrast supplies a confidence gate. Thresholds live in `FlowTestConfig` and are emitted in the result for traceability.

## Honest limitations

- **No real-video validation has been performed.** Real captures are required to establish MAE, level agreement, failure rates, phone/lighting robustness, and appropriate thresholds. Do not quote the synthetic MAE as product accuracy.
- Transparent/colorless liquids, reflections, curved menisci, foam, labels, patterned backgrounds, rolling shutter, autofocus, camera motion, perspective mismatch between card and syringe, and nonstandard syringes may defeat these heuristics.
- The printed rectangle supplies scale but not identity or print-size authentication. A future capture app should use a versioned coded fiducial and camera-pose checks.
- The bubble/lump and occlusion checks are warning heuristics, not reliable material-property detectors. Human inspection and repeat testing remain necessary.
- Volume is inferred linearly along the calibrated barrel. The method assumes the detected endpoints coincide with the 0 and 10 mL marks.
- L4 is reported only when release is observed but flow stays below the configured detectable-motion threshold. L4 still requires the other official IDDSI tests where applicable.
- This tool does not determine suitability for a person, aspiration risk, or whether anything is safe to consume. It is not clinically validated, not medical-grade, and not endorsed by IDDSI.

Before any public performance claim, validate on physically tested real videos with locked external captures and report failures and abstentions as required by the project runboard.
