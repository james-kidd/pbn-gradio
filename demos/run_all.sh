#!/usr/bin/env bash
# Execute every notebook headlessly, in place, and report pass/fail.
# Used to confirm the notebooks still run against the current toolkit.
set -uo pipefail
cd "$(dirname "$0")"

failed=0
for nb in notebooks/*.ipynb; do
    if python -m nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=600 "$nb" >/dev/null 2>&1; then
        echo "  ok    $nb"
    else
        echo "  FAIL  $nb"
        failed=1
    fi
done

[ "$failed" -eq 0 ] && echo "all notebooks executed" || echo "at least one notebook failed"
exit "$failed"
