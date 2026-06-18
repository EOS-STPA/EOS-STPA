@echo off
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
python scripts\run_validation.py --ontology ontology\eos_stpa_avx.ttl --query queries\delayed_red_light_validation.rq --outdir results
if errorlevel 1 exit /b 1
