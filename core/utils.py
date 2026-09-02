import os
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict, deque

def has_path(start: str, end: str, dag_children: Dict[str, List[str]]) -> bool:
    """
    Check to see if a path from start to end already exists in a DAG
    """
    visited = set()
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == end:
            return True
        if current in visited:
            continue
        visited.add(current)
        for nbr in dag_children.get(current, []):
            if nbr not in visited:
                queue.append(nbr)
    return False

def find_directed_paths(
    start: str,
    end: str,
    dag_children: Dict[str, List[str]],
    max_paths: int,
    max_dfs_steps: int = 200_000,
) -> List[List[str]]:
    """
    Enumerate up to max_paths+1 simple directed paths start -> ... -> end (inclusive).
    Returns the list (capped at max_paths+1 entries so the caller can detect overflow
    via len(result) > max_paths). Empty list means no path exists.

    max_dfs_steps bounds compute on dense graphs where the simple-path count is
    combinatorial. On budget exhaustion the result is padded past max_paths so callers
    see it as overflow and fall through their existing "skip arbitration" branch.
    """
    results: List[List[str]] = []
    visited: set = set()
    path: List[str] = []
    cap = max_paths + 1
    steps = [0]
    budget_exhausted = [False]

    def dfs(node: str) -> None:
        if budget_exhausted[0] or len(results) >= cap:
            return
        steps[0] += 1
        if steps[0] > max_dfs_steps:
            budget_exhausted[0] = True
            return
        path.append(node)
        visited.add(node)
        if node == end:
            results.append(list(path))
        else:
            for nbr in dag_children.get(node, []):
                if nbr in visited:
                    continue
                dfs(nbr)
                if budget_exhausted[0] or len(results) >= cap:
                    break
        path.pop()
        visited.remove(node)

    dfs(start)
    if budget_exhausted[0]:
        while len(results) < cap:
            results.append([start, end])
    return results

def extract_filename(path: str) -> str:
    """
    Extract a filename (without extension) from a filesystem path.

    Args:
        path: String path to a file or directory.

    Returns:
        Filename without its extension, or last directory name if path is a directory.
    """
    path = path.replace('\\', '/').rstrip('/')
    parts = path.split('/')
    last_part = parts[-1]
    # Check if it's a file (contains a dot)
    if '.' in last_part:
        return '.'.join(last_part.split('.')[:-1])
    else:
        return last_part

def numpy_to_python(obj: Any) -> Union[list, int, float, bool, None, Any]:
    """
    Convert NumPy scalar and array types into native Python types.

    Args:
        obj: Object potentially a NumPy scalar or array.

    Returns:
        Converted Python native type, or original object if no conversion needed.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.void):
        return None
    return obj

def get_cols_with_prefix(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    List column names from a DataFrame that start with a given prefix.

    Args:
        df: pandas DataFrame to inspect.
        prefix: String prefix to filter column names.

    Returns:
        List of column names beginning with the prefix.
    """
    return [c for c in df.columns if c.startswith(prefix)]

def jaccard_similarity(list1: List[Any], list2: List[Any]) -> float:
    """
    Compute Jaccard similarity between two lists.

    Args:
        list1: First list of hashable items.
        list2: Second list of hashable items.

    Returns:
        Jaccard similarity (intersection over union).
    """
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def mb_jaccard_similarity(
    node_markov_blankets_1: Dict[str, List[Any]],
    node_markov_blankets_2: Dict[str, List[Any]]
) -> float:
    """
    Compute mean Jaccard similarity across corresponding Markov blankets.

    Args:
        node_markov_blankets_1: Mapping node -> list of nodes in its Markov blanket.
        node_markov_blankets_2: Same as above for a second model.

    Returns:
        Average Jaccard similarity across nodes present in both maps.
    """
    similarities: List[float] = []
    for node, mb1 in node_markov_blankets_1.items():
        if node in node_markov_blankets_2:
            mb2 = node_markov_blankets_2[node]
            similarities.append(jaccard_similarity(mb1, mb2))
    return float(np.mean(similarities))
