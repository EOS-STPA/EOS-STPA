#!/usr/bin/env python3
"""Reproducible EOS-STPA delayed-red-light ontology validation.

The script loads the machine-readable OWL/SWRL ontology, performs a small
forward-chaining pass over the SWRL subset used by the validation case,
materializes basic RDFS subclass entailments, executes the SPARQL query, and
writes an inference trace and result files.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from rdflib import BNode, Graph, Namespace, RDF, RDFS, URIRef
from rdflib.collection import Collection

SWRL = Namespace("http://www.w3.org/2003/11/swrl#")

Atom = Tuple[str, URIRef, URIRef, URIRef | None]
Binding = Dict[URIRef, URIRef]


def is_variable(graph: Graph, term: URIRef) -> bool:
    return (term, RDF.type, SWRL.Variable) in graph


def rdf_list(graph: Graph, head: URIRef | BNode) -> List[URIRef | BNode]:
    return list(Collection(graph, head))


def parse_atom(graph: Graph, atom_node: URIRef | BNode) -> Atom:
    atom_types = set(graph.objects(atom_node, RDF.type))
    if SWRL.ClassAtom in atom_types:
        predicate = graph.value(atom_node, SWRL.classPredicate)
        arg1 = graph.value(atom_node, SWRL.argument1)
        if not isinstance(predicate, URIRef) or not isinstance(arg1, URIRef):
            raise ValueError(f"Malformed SWRL class atom: {atom_node}")
        return ("class", predicate, arg1, None)
    if SWRL.IndividualPropertyAtom in atom_types:
        predicate = graph.value(atom_node, SWRL.propertyPredicate)
        arg1 = graph.value(atom_node, SWRL.argument1)
        arg2 = graph.value(atom_node, SWRL.argument2)
        if not all(isinstance(x, URIRef) for x in (predicate, arg1, arg2)):
            raise ValueError(f"Malformed SWRL property atom: {atom_node}")
        return ("property", predicate, arg1, arg2)
    raise ValueError(f"Unsupported SWRL atom type at {atom_node}")


def parse_rules(graph: Graph) -> List[Tuple[URIRef, List[Atom], List[Atom]]]:
    rules = []
    for rule in sorted(graph.subjects(RDF.type, SWRL.Imp), key=str):
        body_head = graph.value(rule, SWRL.body)
        head_head = graph.value(rule, SWRL.head)
        if body_head is None or head_head is None:
            continue
        body = [parse_atom(graph, n) for n in rdf_list(graph, body_head)]
        head = [parse_atom(graph, n) for n in rdf_list(graph, head_head)]
        rules.append((rule, body, head))
    return rules


def resolve(term: URIRef, binding: Binding, graph: Graph) -> URIRef | None:
    if is_variable(graph, term):
        return binding.get(term)
    return term


def extend_binding(term: URIRef, value: URIRef, binding: Binding, graph: Graph) -> Binding | None:
    if not is_variable(graph, term):
        return binding if term == value else None
    if term in binding and binding[term] != value:
        return None
    new_binding = dict(binding)
    new_binding[term] = value
    return new_binding


def match_atom(graph: Graph, atom: Atom, binding: Binding) -> Iterable[Binding]:
    kind, predicate, arg1, arg2 = atom
    if kind == "class":
        bound_subject = resolve(arg1, binding, graph)
        pattern_subject = bound_subject if bound_subject is not None else None
        for subject, _, _ in graph.triples((pattern_subject, RDF.type, predicate)):
            if not isinstance(subject, URIRef):
                continue
            updated = extend_binding(arg1, subject, binding, graph)
            if updated is not None:
                yield updated
        return

    assert arg2 is not None
    bound_s = resolve(arg1, binding, graph)
    bound_o = resolve(arg2, binding, graph)
    for subject, _, obj in graph.triples((bound_s, predicate, bound_o)):
        if not isinstance(subject, URIRef) or not isinstance(obj, URIRef):
            continue
        updated = extend_binding(arg1, subject, binding, graph)
        if updated is None:
            continue
        updated = extend_binding(arg2, obj, updated, graph)
        if updated is not None:
            yield updated


def body_bindings(graph: Graph, atoms: Sequence[Atom]) -> List[Binding]:
    bindings: List[Binding] = [{}]
    for atom in atoms:
        next_bindings: List[Binding] = []
        for binding in bindings:
            next_bindings.extend(match_atom(graph, atom, binding))
        bindings = next_bindings
        if not bindings:
            break
    return bindings


def instantiate_head(graph: Graph, atom: Atom, binding: Binding) -> Tuple[URIRef, URIRef, URIRef]:
    kind, predicate, arg1, arg2 = atom
    subject = resolve(arg1, binding, graph)
    if subject is None:
        raise ValueError("Unbound SWRL head variable")
    if kind == "class":
        return subject, RDF.type, predicate
    assert arg2 is not None
    obj = resolve(arg2, binding, graph)
    if obj is None:
        raise ValueError("Unbound SWRL head variable")
    return subject, predicate, obj


def materialize_subclasses(graph: Graph, trace: List[dict]) -> int:
    added = 0
    changed = True
    while changed:
        changed = False
        subclass_pairs = list(graph.triples((None, RDFS.subClassOf, None)))
        type_pairs = list(graph.triples((None, RDF.type, None)))
        for instance, _, cls in type_pairs:
            for sub, _, parent in subclass_pairs:
                if not isinstance(parent, URIRef):
                    continue
                if cls == sub and (instance, RDF.type, parent) not in graph:
                    graph.add((instance, RDF.type, parent))
                    trace.append({
                        "step": len(trace) + 1,
                        "source": "RDFS subclass entailment",
                        "subject": str(instance),
                        "predicate": str(RDF.type),
                        "object": str(parent),
                    })
                    added += 1
                    changed = True
    return added


def run_rules(graph: Graph, trace: List[dict]) -> int:
    rules = parse_rules(graph)
    added = 0
    changed = True
    while changed:
        changed = False
        for rule, body, head in rules:
            for binding in body_bindings(graph, body):
                for atom in head:
                    triple = instantiate_head(graph, atom, binding)
                    if triple not in graph:
                        graph.add(triple)
                        trace.append({
                            "step": len(trace) + 1,
                            "source": str(rule),
                            "subject": str(triple[0]),
                            "predicate": str(triple[1]),
                            "object": str(triple[2]),
                        })
                        added += 1
                        changed = True
        if materialize_subclasses(graph, trace):
            changed = True
    return added


def write_query_results(graph: Graph, query_path: Path, out_csv: Path) -> tuple[int, list[str], list[list[str]]]:
    query_text = query_path.read_text(encoding="utf-8")
    results = graph.query(query_text)
    columns = [str(v) for v in results.vars]
    rows = []
    for row in results:
        rows.append([str(value) if value is not None else "" for value in row])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return len(rows), columns, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    graph = Graph()
    graph.parse(args.ontology, format="turtle")
    asserted_count = len(graph)

    trace: List[dict] = []
    materialize_subclasses(graph, trace)
    inferred_count = run_rules(graph, trace)

    inferred_graph_path = args.outdir / "inferred_graph.ttl"
    graph.serialize(inferred_graph_path, format="turtle")

    trace_path = args.outdir / "reasoning_trace.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "source", "subject", "predicate", "object"])
        writer.writeheader()
        writer.writerows(trace)

    result_path = args.outdir / "query_results.csv"
    result_count, result_columns, result_rows = write_query_results(graph, args.query, result_path)

    expected_columns = [
        "saic", "hazard", "controlSubject", "controlObject",
        "conditionObject", "faultMode", "ssc", "sscLabel",
    ]
    expected_row = [
        "https://EOS-STPA.github.io/EOS-STPA/ontology#Input_001",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#CollisionRelatedHazard",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#VehicleAction",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#InBraking",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#RedLight",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#Late",
        "https://EOS-STPA.github.io/EOS-STPA/ontology#SSC_DelayedRedLight_001",
        "Vehicle action shall be in braking when traffic infrastructure provides late red light",
    ]
    validation_passed = (
        result_count == 1
        and result_columns == expected_columns
        and result_rows == [expected_row]
    )

    summary = {
        "ontology_file": "ontology/eos_stpa_avx.ttl",
        "query_file": "queries/delayed_red_light_validation.rq",
        "asserted_triple_count": asserted_count,
        "materialized_graph_triple_count": len(graph),
        "new_inferences_recorded": len(trace),
        "swrl_head_triples_added": inferred_count,
        "query_result_count": result_count,
        "validation_passed": validation_passed,
    }
    (args.outdir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    if not validation_passed:
        raise SystemExit("Validation failed: query result did not match the expected SAIC-hazard-SSC trace")


if __name__ == "__main__":
    main()
