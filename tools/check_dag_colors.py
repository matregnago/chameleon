#!/usr/bin/env python3
#
# @file check_dag_colors.py
#
# @copyright 2026-2026 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria,
#                      Univ. Bordeaux. All rights reserved.
#
# @version 1.4.0
# @author Mathieu Faverge
# @date 2026-08-02
#
"""Check that DAG color mappings cover generated tasks and algorithms."""

from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TASK_COLORS = ROOT / "control" / "dag_task_colors.h"
DAG_COLORS = ROOT / "control" / "dag_colors.h"
STARPU_CODELETS = ROOT / "runtime" / "starpu" / "codelets"
QUARK_DAG = ROOT / "runtime" / "quark" / "include" / "core_blas_dag.h"
COMPUTE = ROOT / "compute"
CODEGEN = (
    ROOT
    / "cmake_modules"
    / "morse_cmake"
    / "modules"
    / "precision_generator"
    / "codegen.py"
)
SUBSTITUTIONS = ROOT / "cmake_modules" / "local_subs.py"

IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
TASK_NAME_DEFINE = re.compile(
    rf"^\s*#define\s+CHAMELEON_DAG_TASK_COLOR_NAME_({IDENTIFIER})\s+([A-Z][A-Z0-9_]*)\s*$",
    re.MULTILINE,
)
TASK_COLOR_DEFINE = re.compile(
    r"^\s*#define\s+CHAMELEON_DAG_TASK_COLOR_([A-Z][A-Z0-9_]*)\s+"
    r"CHAMELEON_DAG_COLOR_([A-Z][A-Z0-9_]*)\s*$",
    re.MULTILINE,
)
ALGORITHM_COLOR_DEFINE = re.compile(
    rf"^\s*#define\s+CHAMELEON_DAG_ALG_COLOR_({IDENTIFIER})\s+"
    r"CHAMELEON_DAG_COLOR_([A-Z][A-Z0-9_]*)\s*$",
    re.MULTILINE,
)
PALETTE_VALUE_DEFINE = re.compile(
    r"^\s*#define\s+CHAMELEON_DAG_COLOR_([A-Z][A-Z0-9_]*)\s+0x([0-9a-fA-F]{6})\s*$",
    re.MULTILINE,
)
PALETTE_STRING_DEFINE = re.compile(
    r'^\s*#define\s+CHAMELEON_DAG_COLOR_([A-Z][A-Z0-9_]*)_STR\s+"#([0-9a-fA-F]{6})"\s*$',
    re.MULTILINE,
)

INSERT_TASK_PARAMS = re.compile(
    rf"INSERT_TASK_COMMON_TASK_PARAMS(?:_NOCB)?\s*\(\s*({IDENTIFIER})"
)
SUBMIT_PARAMS = re.compile(
    rf"INSERT_TASK_COMMON_PARAMETERS(?:_CLNULL)?\s*\(\s*({IDENTIFIER})"
)
SUBMIT_EXTENDED_PARAMS = re.compile(
    rf"INSERT_TASK_COMMON_PARAMETERS_EXTENDED\s*\(\s*{IDENTIFIER}\s*,\s*({IDENTIFIER})"
)
QUARK_TASK_CLASS = re.compile(
    r'DAG_SET_PROPERTIES\s*\(\s*"[^"]+"\s*,\s*([A-Z][A-Z0-9_]*)\s*\)'
)
ALGORITHM_COLOR_USE = re.compile(
    rf"CHAMELEON_DAG_COLOR_ALGORITHM\s*\(\s*({IDENTIFIER})\s*\)"
)


def duplicate_keys(pairs):
    """Return duplicate keys from a sequence of key/value pairs."""
    seen = set()
    duplicates = set()
    for key, _ in pairs:
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return duplicates


def generate_starpu_codelets(output_dir):
    """Generate every precision variant used to validate StarPU task names."""
    sources = []
    for source in sorted(STARPU_CODELETS.glob("*.c")):
        if "@precisions" in source.read_text(encoding="utf-8"):
            sources.append(str(source.relative_to(ROOT)))

    command = [
        sys.executable,
        str(CODEGEN),
        "-g",
        "-f",
        " ".join(sources),
        "-p",
        "s d c z ds zc",
        "-s",
        str(ROOT),
        "-P",
        str(output_dir),
        "-D",
        str(SUBSTITUTIONS),
    ]
    subprocess.run(command, check=True)


def report_missing(errors, label, used, defined):
    """Append one error per used identifier absent from its definition set."""
    for name in sorted(used - defined):
        errors.append(f"{label}: missing {name}")


def main():
    errors = []
    task_text = TASK_COLORS.read_text(encoding="utf-8")
    dag_text = DAG_COLORS.read_text(encoding="utf-8")

    task_name_pairs = TASK_NAME_DEFINE.findall(task_text)
    task_color_pairs = TASK_COLOR_DEFINE.findall(task_text)
    algorithm_color_pairs = ALGORITHM_COLOR_DEFINE.findall(dag_text)
    palette_value_pairs = PALETTE_VALUE_DEFINE.findall(dag_text)
    palette_string_pairs = PALETTE_STRING_DEFINE.findall(dag_text)

    for label, pairs in (
        ("task name", task_name_pairs),
        ("task color", task_color_pairs),
        ("algorithm color", algorithm_color_pairs),
        ("palette value", palette_value_pairs),
        ("palette string", palette_string_pairs),
    ):
        for name in sorted(duplicate_keys(pairs)):
            errors.append(f"{label}: duplicate {name}")

    task_names = dict(task_name_pairs)
    task_colors = dict(task_color_pairs)
    algorithm_colors = dict(algorithm_color_pairs)
    palette_values = {name: value.lower() for name, value in palette_value_pairs}
    palette_strings = {name: value.lower() for name, value in palette_string_pairs}

    report_missing(errors, "task class", set(task_names.values()), set(task_colors))
    report_missing(errors, "task palette", set(task_colors.values()), set(palette_values))
    report_missing(errors, "algorithm palette", set(algorithm_colors.values()), set(palette_values))

    report_missing(errors, "palette string", set(palette_values), set(palette_strings))
    report_missing(errors, "palette value", set(palette_strings), set(palette_values))
    for name in sorted(set(palette_values) & set(palette_strings)):
        if palette_values[name] != palette_strings[name]:
            errors.append(
                f"palette mismatch: {name} is 0x{palette_values[name]} "
                f"and #{palette_strings[name]}"
            )

    with tempfile.TemporaryDirectory(prefix="chameleon-dag-colors-") as tmpdir:
        generated_dir = Path(tmpdir)
        generate_starpu_codelets(generated_dir)

        starpu_names = set()
        codelets = list(STARPU_CODELETS.glob("*.c")) + list(generated_dir.glob("*.c"))
        for codelet in codelets:
            text = codelet.read_text(encoding="utf-8")
            starpu_names.update(INSERT_TASK_PARAMS.findall(text))
            starpu_names.update(SUBMIT_PARAMS.findall(text))
            starpu_names.update(SUBMIT_EXTENDED_PARAMS.findall(text))

    report_missing(errors, "StarPU task name", starpu_names, set(task_names))

    quark_text = QUARK_DAG.read_text(encoding="utf-8")
    report_missing(
        errors,
        "Quark task class",
        set(QUARK_TASK_CLASS.findall(quark_text)),
        set(task_colors),
    )

    algorithm_uses = set()
    for source in sorted(COMPUTE.glob("*.c")):
        text = source.read_text(encoding="utf-8")
        if (
            "RUNTIME_options_init" in text
            and "RUNTIME_options_set_taskcolor" not in text
        ):
            errors.append(
                f"algorithm color: missing assignment in {source.relative_to(ROOT)}"
            )
        algorithm_uses.update(ALGORITHM_COLOR_USE.findall(text))
    report_missing(errors, "algorithm color", algorithm_uses, set(algorithm_colors))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print("DAG color mappings: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
