"""Synthetic benchmark for the IDDSI Flow-Test grader.

Deterministic, varied synthetic-video benchmark over a documented parameter
grid (residual volume, camera tilt, bubbles, lighting). Reuses
``flowtest.synthetic.generate_video``/``SyntheticSpec`` for video synthesis and
``flowtest.pipeline.grade_video`` for grading; nothing is duplicated.

Abstention uses the grader's real mechanism: ``GradeResult.status ==
"abstain"`` with codes in ``GradeResult.abstention_reasons`` (see
``flowtest.abstention``).

Run from the ``flowtest`` package directory::

    python -m flowtest.bench.run_bench --n 100 --out results_n100

This module deliberately does NOT import ``.run_bench`` here: the CLI is
executed via ``python -m flowtest.bench.run_bench``, and pre-importing the
submodule from the package ``__init__`` triggers a RuntimeWarning and double
execution. Import the submodule directly instead::

    from flowtest.bench.run_bench import GRID, build_grid, run_bench

Synthetic fixtures are pipeline-development checks only; this benchmark is
NOT evidence of performance on real liquids or real phone videos, and it is
not a safety benchmark.
"""

__all__ = ["run_bench"]
