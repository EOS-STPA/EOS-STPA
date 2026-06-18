#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -r requirements.txt
python3 scripts/run_validation.py \
  --ontology ontology/eos_stpa_avx.ttl \
  --query queries/delayed_red_light_validation.rq \
  --outdir results
