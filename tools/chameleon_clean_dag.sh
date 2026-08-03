#!/usr/bin/env sh

set -eu

usage()
{
    cat <<EOF
Usage:
  $0 [options] -- ./chameleon_dtesting -o trtri -n 100 -b 10 [args...]
  $0 [options] --input-dot dag.dot

Options:
  -o, --output NAME       Output basename (default: chameleon-dag)
  -d, --output-dir DIR    Directory for traces and generated files (default: .)
      --input-dot FILE    Clean an existing StarPU dag.dot instead of running a command
      --kernel-only       Keep only the kernel name in task labels
      --color-mode MODE   Set CHAMELEON_DAG_COLOR_MODE: algorithm or task
      --color-algorithm   Alias for --color-mode algorithm
      --color-task        Alias for --color-mode task
      --no-splitsub       Do not add --splitsub automatically
      --keep              Keep StarPU intermediate files
  -h, --help              Show this help

Generated files:
  NAME.raw.dot            DOT produced by starpu_fxt_tool
  NAME.dot                Cleaned DOT
  NAME.pdf                Cleaned PDF
EOF
}

outbase=chameleon-dag
outdir=.
input_dot=
kernel_only=0
dag_color_mode=
add_splitsub=1
keep=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--output)
            outbase=$2
            shift 2
            ;;
        -d|--output-dir)
            outdir=$2
            shift 2
            ;;
        --input-dot)
            input_dot=$2
            shift 2
            ;;
        --kernel-only)
            kernel_only=1
            shift
            ;;
        --color-mode)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --color-mode" >&2
                echo "Expected: algorithm or task" >&2
                exit 2
            fi
            case "$2" in
                algorithm|algo|alg|ChamDagColorAlgorithm)
                    dag_color_mode=ChamDagColorAlgorithm
                    ;;
                task|tasks|ChamDagColorTask)
                    dag_color_mode=ChamDagColorTask
                    ;;
                *)
                    echo "Invalid --color-mode value: $2" >&2
                    echo "Expected: algorithm or task" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --color-algorithm)
            dag_color_mode=ChamDagColorAlgorithm
            shift
            ;;
        --color-task)
            dag_color_mode=ChamDagColorTask
            shift
            ;;
        --no-splitsub)
            add_splitsub=0
            shift
            ;;
        --keep)
            keep=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "$outdir"
outdir=$(cd "$outdir" && pwd)
raw_dot=$outdir/$outbase.raw.dot
clean_dot=$outdir/$outbase.dot
pdf=$outdir/$outbase.pdf

have_arg()
{
    wanted=$1
    shift
    for arg in "$@"; do
        [ "$arg" = "$wanted" ] && return 0
    done
    return 1
}

if [ -n "$input_dot" ]; then
    cp "$input_dot" "$raw_dot"
else
    if [ "$#" -eq 0 ]; then
        usage >&2
        exit 2
    fi

    if ! have_arg --trace "$@" && ! have_arg -T "$@"; then
        set -- "$@" --trace
    fi
    if [ "$add_splitsub" -eq 1 ] && ! have_arg --splitsub "$@" && ! have_arg -S "$@"; then
        set -- "$@" --splitsub
    fi

    rm -f "$outdir"/prof_file_* "$outdir"/dag.dot

    (
        export STARPU_FXT_PREFIX=$outdir
        export STARPU_FXT_TRACE=1
        export STARPU_FXT=1
        if [ -n "$dag_color_mode" ]; then
            export CHAMELEON_DAG_COLOR_MODE=$dag_color_mode
        fi
        "$@"
    )

    prof_files=$(find "$outdir" -maxdepth 1 -type f -name 'prof_file_*' | sort)
    if [ -z "$prof_files" ]; then
        echo "No StarPU FxT prof_file_* generated in $outdir" >&2
        exit 1
    fi

    set -- starpu_fxt_tool
    for prof_file in $prof_files; do
        set -- "$@" -i "$prof_file"
    done

    (
        cd "$outdir"
        "$@"
    )

    if [ ! -f "$outdir/dag.dot" ]; then
        echo "starpu_fxt_tool did not generate $outdir/dag.dot" >&2
        exit 1
    fi

    mv "$outdir/dag.dot" "$raw_dot"
fi

python3 - "$raw_dot" "$clean_dot" "$kernel_only" <<'PY'
import re
import sys
from collections import defaultdict, deque

raw_dot, clean_dot, kernel_only = sys.argv[1], sys.argv[2], sys.argv[3] == "1"

edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*(?:\[(.*)\])?\s*;?\s*$')
node_re = re.compile(r'^\s*"([^"]+)"\s*\[(.*)\]\s*;?\s*$')
label_re = re.compile(r'label\s*=\s*"((?:\\"|[^"])*)"')

drop_labels = {
    "",
    "_starpu_data_acquire_cb_pre",
    "_starpu_data_acquire_cb_release",
    "_starpu_data_acquire_cb",
    "chameleon_flush",
}

edges = []
nodes = {}
node_order = []

with open(raw_dot, "r", encoding="utf-8") as f:
    for line in f:
        if "href=" in line or re.search(r"\brankdir\b", line):
            continue

        match = edge_re.match(line)
        if match:
            src, dst, attrs = match.groups()
            edges.append((src, dst, (attrs or "").strip(), "normal"))
            continue

        match = node_re.match(line)
        if not match:
            continue

        node, attrs = match.groups()
        label_match = label_re.search(attrs)
        if label_match is None:
            continue

        if node not in nodes:
            node_order.append(node)
        nodes[node] = attrs.strip()

def label_of(node):
    attrs = nodes.get(node, "")
    match = label_re.search(attrs)
    return match.group(1) if match else None

def dot_escape(value):
    return value.replace("\\", "\\\\").replace('"', '\\"')

def kernel_label(label):
    return re.sub(r'\s*\(.*\)\s*$', "", label).strip()

def maybe_shorten_label(attrs):
    if not kernel_only:
        return attrs

    match = label_re.search(attrs)
    if match is None:
        return attrs

    label = match.group(1)
    short_label = dot_escape(kernel_label(label))
    return attrs[:match.start(1)] + short_label + attrs[match.end(1):]

sync_nodes = {node for node in nodes if label_of(node) == "_starpu_sync_task"}
drop_nodes = {
    node
    for node in nodes
    if label_of(node) in drop_labels
}
removed_nodes = drop_nodes | sync_nodes

kept_nodes = [
    node for node in node_order
    if node in nodes and node not in removed_nodes
]
kept = set(kept_nodes)

succs = defaultdict(list)
for src, dst, attrs, kind in edges:
    succs[src].append((dst, attrs))

def edge_attrs(kind, attrs=""):
    if kind == "anti":
        return 'color="red", fontcolor="red", penwidth=2'
    return attrs

clean_edges = []
for start in kept_nodes:
    queue = deque()
    for dst, attrs in succs[start]:
        traversed_sync = dst in sync_nodes
        if dst in kept:
            clean_edges.append((start, dst, attrs, "normal"))
        else:
            queue.append((dst, traversed_sync))

    seen_removed = set()
    while queue:
        node, traversed_sync = queue.popleft()
        state = (node, traversed_sync)
        if state in seen_removed:
            continue
        seen_removed.add(state)

        if node in kept:
            kind = "anti" if traversed_sync else "normal"
            clean_edges.append((start, node, edge_attrs(kind), kind))
            continue

        for dst, attrs in succs[node]:
            queue.append((dst, traversed_sync or dst in sync_nodes))

seen_edges = set()
dedup_edges = []
for edge in clean_edges:
    key = edge
    if key in seen_edges:
        continue
    seen_edges.add(key)
    dedup_edges.append(edge)
clean_edges = dedup_edges

succ = defaultdict(list)
indeg = {node: 0 for node in kept_nodes}
for src, dst, attrs, kind in clean_edges:
    if src not in kept or dst not in kept:
        continue
    succ[src].append(dst)
    indeg[dst] += 1

level = {node: 0 for node in kept_nodes}
queue = deque([node for node in kept_nodes if indeg[node] == 0])
visited = 0
while queue:
    node = queue.popleft()
    visited += 1
    for dst in succ[node]:
        if level[dst] < level[node] + 1:
            level[dst] = level[node] + 1
        indeg[dst] -= 1
        if indeg[dst] == 0:
            queue.append(dst)

if visited != len(kept_nodes):
    remaining = [node for node in kept_nodes if indeg[node] > 0]
    for node in remaining:
        level[node] = max(level.values(), default=0) + 1

by_level = defaultdict(list)
for node in kept_nodes:
    by_level[level[node]].append(node)

max_level = max(level.values(), default=0)

with open(clean_dot, "w", encoding="utf-8") as out:
    out.write("digraph G {\n")
    out.write("\tcolor=white;\n")
    out.write("\trankdir=TB;\n")
    out.write("\tedge [arrowsize=0.7];\n")
    out.write("\tsubgraph cluster_0 {\n")
    out.write("\t\tcolor=black;\n")

    for i in range(max_level + 1):
        out.write(f'\t\t"_rank_{i}" [shape=circle, label="{i}", color="black", fontcolor="black"];\n')

    for i in range(max_level):
        out.write(f'\t\t"_rank_{i}" -> "_rank_{i + 1}" [style=invis, weight=1000];\n')

#    for i in range(max_level + 1):
#        tasks = by_level.get(i, [])
#        if tasks:
#            out.write(f'\t\t"_rank_{i}" -> "{tasks[0]}" [style=invis, weight=1000];\n')

    for src, dst, attrs, kind in clean_edges:
        if src not in kept or dst not in kept:
            continue
        if attrs:
            out.write(f'\t\t"{src}" -> "{dst}" [{attrs}];\n')
        else:
            out.write(f'\t\t"{src}" -> "{dst}";\n')

    for node in kept_nodes:
        out.write(f'\t\t"{node}" [{maybe_shorten_label(nodes[node])}];\n')

    for i in range(max_level + 1):
        tasks = " ".join(f'"{node}";' for node in by_level.get(i, []))
        out.write(f'\t\t{{ rank=same; "_rank_{i}"; {tasks} }}\n')

    out.write("\t}\n")
    out.write("}\n")
PY

dot -Tpdf -o "$pdf" "$clean_dot"

if [ "$keep" -eq 0 ] && [ -z "$input_dot" ]; then
    rm -f "$outdir"/paje.trace "$outdir"/*.rec
fi

echo "Raw DOT:     $raw_dot"
echo "Clean DOT:   $clean_dot"
echo "Clean PDF:   $pdf"
