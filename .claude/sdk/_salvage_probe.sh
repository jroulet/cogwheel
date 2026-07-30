#!/bin/sh
# Driver scratch probe: run the four 1b-affected suites and record the result.
# Not part of the pipeline; safe to delete. Exists so the run survives the
# interactive shell's 120s cap by detaching cleanly.
cd "$(git rev-parse --show-toplevel)" || exit 1
PY="$(conda info --base)/envs/${SDK_CONDA_ENV:-cogwheel-newlal}/bin/python"
"$PY" -m pytest -q -p no:cacheprovider \
    cogwheel/tests/test_lensing_caustic_cusps.py \
    cogwheel/tests/test_lensing_surrogate.py \
    cogwheel/tests/test_lensing_surrogate_training.py \
    cogwheel/tests/test_lensing_exterior_admission.py \
    > /tmp/salvage_pytest.log 2>&1
echo "EXIT=$? $(date +%T)" >> /tmp/salvage_pytest.log
