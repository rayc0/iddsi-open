# Flow-Test synthetic bench (n=100)

synthetic pipeline-development benchmark only; not real-video validation and not a safety benchmark

Grid: {"residual_ml": [0.5, 2.5, 4.25, 6.0, 8.4, 10.0], "tilt_deg": [0.0, 2.0, 4.0], "bubbles": [false, true], "lighting_factor": [1.0, 0.75, 1.15]}; fixed: {"fps": 8.0, "width": 640, "height": 480}; case i uses grid combo i % len(grid) with SyntheticSpec seed 1000 + i

| n | graded | abstained | abstention rate | MAE mL (all estimates) | MAE mL (graded only) | n with estimate |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 16 | 84 | 0.84 | 0.6224 | 0.0651 | 97 |

## Per bucket: volume_bucket
| bucket | n | graded | abstained | abstention rate | MAE mL | MAE mL (graded only) |
| --- | --- | --- | --- | --- | --- | --- |
| far_from_boundary | 46 | 16 | 30 | 0.6522 | 0.6321 | 0.0651 |
| near_boundary | 54 | 0 | 54 | 1.0 | 0.614 | None |

## Per bucket: residual_ml
| bucket | n | graded | abstained | abstention rate | MAE mL | MAE mL (graded only) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5 | 18 | 0 | 18 | 1.0 | 1.2342 | None |
| 10.0 | 10 | 4 | 6 | 0.6 | 0.0 | 0.0 |
| 2.5 | 18 | 6 | 12 | 0.6667 | 0.9688 | 0.1273 |
| 4.25 | 18 | 0 | 18 | 1.0 | 0.4252 | None |
| 6.0 | 18 | 6 | 12 | 0.6667 | 0.6652 | 0.0463 |
| 8.4 | 18 | 0 | 18 | 1.0 | 0.1617 | None |

## Per bucket: tilt_deg
| bucket | n | graded | abstained | abstention rate | MAE mL | MAE mL (graded only) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 36 | 6 | 30 | 0.8333 | 0.0888 | 0.0575 |
| 2.0 | 34 | 6 | 28 | 0.8235 | 1.6221 | 0.0548 |
| 4.0 | 30 | 4 | 26 | 0.8667 | 0.2297 | 0.092 |

## Per bucket: bubbles
| bucket | n | graded | abstained | abstention rate | MAE mL | MAE mL (graded only) |
| --- | --- | --- | --- | --- | --- | --- |
| False | 51 | 16 | 35 | 0.6863 | 0.6111 | 0.0651 |
| True | 49 | 0 | 49 | 1.0 | 0.6349 | None |

## Per bucket: lighting_factor
| bucket | n | graded | abstained | abstention rate | MAE mL | MAE mL (graded only) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.75 | 33 | 0 | 33 | 1.0 | 1.6997 | None |
| 1.0 | 34 | 8 | 26 | 0.7647 | 0.0953 | 0.0728 |
| 1.15 | 33 | 8 | 25 | 0.7576 | 0.0887 | 0.0575 |

## Confusion (rows = ground truth, columns = prediction; last column = grader abstained)
| gt \ pred | L0 | L1 | L2 | L3 | L4 | abstain |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | 0 | 0 | 0 | 0 | 0 | 18 |
| L1 | 0 | 6 | 0 | 0 | 0 | 12 |
| L2 | 0 | 0 | 6 | 0 | 0 | 30 |
| L3 | 0 | 0 | 0 | 0 | 0 | 18 |
| L4 | 0 | 0 | 0 | 0 | 4 | 6 |

## Abstention reasons
| code | count |
| --- | --- |
| bubbles_or_lumps_detected | 43 |
| insufficient_duration | 1 |
| meniscus_unclear | 10 |
| near_level_boundary | 47 |
| syringe_not_detected | 2 |
| wrong_or_unverified_syringe | 33 |
