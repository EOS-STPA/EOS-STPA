# EOS-STPA Ontology Reasoning Validation

This repository contains supplementary artifacts for the EOS-STPA study, including:

* an executable ontology reasoning validation for the delayed red-light case;
* a car-following ontology/modeling project archive; and
* CARLA-based PPO training code for the STPA and EOS-STPA strategies.

The executable ontology validation materials are organized in the `ontology/`, `queries/`, `scripts/`, `results/`, and `docs/` directories. The car-following ontology/modeling project is provided in `Ontological structure.zip`, and the CARLA training implementation is provided in `EOS_STPA_CARLA_training.py`.


## Public repository

The public repository is available at:

```text
https://github.com/EOS-STPA/EOS-STPA
```

## Repository structure

```text
.
├── ontology/
│   └── eos_stpa_avx.ttl
├── queries/
│   └── delayed_red_light_validation.rq
├── scripts/
│   ├── run_validation.py
│   └── configure_namespace.py
├── results/
│   ├── inferred_graph.ttl
│   ├── query_results.csv
│   ├── reasoning_trace.csv
│   └── validation_summary.json
├── docs/
│   ├── index.html
│   └── ontology/
│       ├── index.html
│       └── eos_stpa_avx.ttl
├── .github/
│   └── workflows/
│       └── validate.yml
├── requirements.txt
├── run_validation.sh
└── run_validation.bat
```

## Requirements

* Python 3.10 or later
* `rdflib==7.5.0`

Install the required dependency with:

```bash
python -m pip install -r requirements.txt
```

## Run the validation

### Linux or macOS

```bash
bash run_validation.sh
```

### Windows

```bat
run_validation.bat
```

### Manual command

```bash
python scripts/run_validation.py \
  --ontology ontology/eos_stpa_avx.ttl \
  --query queries/delayed_red_light_validation.rq \
  --outdir results
```

A successful run prints a JSON summary and writes:

```json
"validation_passed": true
```

to:

```text
results/validation_summary.json
```

## Expected validation results

The supplied validation case produces the following results:

| Item                                        | Expected value |
| ------------------------------------------- | -------------: |
| Loaded ontology, rule, and instance triples |            234 |
| Materialized graph triples                  |            246 |
| Recorded new entailments                    |             12 |
| SWRL-head triples added                     |              6 |
| SPARQL result rows                          |              1 |
| Validation status                           |         `true` |

The runner validates both the number of returned rows and the exact bindings of all variables selected by the SPARQL query.

## Validation workflow

The executable validation performs the following steps:

1. Loads the ontology, SWRL rules, and instantiated scenario facts from the Turtle/OWL file.
2. Materializes basic RDFS subclass entailments.
3. Executes the deterministic SWRL rule subset required for the delayed red-light case.
4. Retrieves the inferred SAIC--hazard--SSC traceability path through a SPARQL query.
5. Verifies the expected query bindings.
6. Writes the inferred graph, query results, validation summary, and triple-level reasoning trace to the `results/` directory.

## Reasoning scope

The validation runner is intentionally lightweight. It performs:

* basic RDFS subclass materialization;
* deterministic forward chaining over the SWRL rule subset encoded for the validation case; and
* SPARQL retrieval of the inferred SAIC--hazard--SSC path.

The runner is not a complete OWL Description Logic reasoner and is not presented as a cross-scenario validation of the entire EOS-STPA ontology.

## Ontology namespace

The executable ontology uses the following namespace:

```text
https://EOS-STPA.github.io/EOS-STPA/ontology#
```

The namespace is used consistently in:

* `ontology/eos_stpa_avx.ttl`;
* `queries/delayed_red_light_validation.rq`;
* `scripts/run_validation.py`;
* the generated reasoning results; and
* the GitHub Pages ontology documentation.

The machine-readable Turtle/OWL ontology is available at:

```text
https://EOS-STPA.github.io/EOS-STPA/ontology/eos_stpa_avx.ttl
```

The ontology documentation page is available at:

```text
https://EOS-STPA.github.io/EOS-STPA/ontology/
```

## Namespace configuration

The namespace has already been configured for the public EOS-STPA repository.

To regenerate the namespace configuration and validation outputs, run:

```bash
python scripts/configure_namespace.py \
  --username EOS-STPA \
  --repository EOS-STPA
```

The configuration script updates the ontology, SPARQL query, validation script, documentation, and generated results, and then reruns the validation.

## GitHub Pages

This repository includes a `docs/` directory for GitHub Pages publication.

To enable GitHub Pages:

1. Open **Settings → Pages** in the GitHub repository.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Select the `main` branch.
4. Select the `/docs` folder.
5. Click **Save** and wait for deployment to complete.

The resulting GitHub Pages site is:

```text
https://EOS-STPA.github.io/EOS-STPA/
```

## Generated outputs

After successful execution, the following files are generated or updated:

* `results/inferred_graph.ttl`: materialized RDF graph;
* `results/query_results.csv`: SPARQL query bindings;
* `results/reasoning_trace.csv`: triple-level inference trace; and
* `results/validation_summary.json`: execution statistics and validation status.

## Reproducibility

The repository includes a GitHub Actions workflow located at:

```text
.github/workflows/validate.yml
```

The workflow installs the required dependency and reruns the validation whenever the repository is updated. A successful workflow run confirms that the executable validation produces the expected results.

## Limitations

This repository provides an executable proof of concept for the delayed red-light case. It does not constitute complete OWL DL reasoning or comprehensive validation across all operational-scenario families addressed by EOS-STPA.
