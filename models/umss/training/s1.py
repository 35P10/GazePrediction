"""Our adapter for executing the original UMSS-S1 training cells.

Local paths, logging, seeds and optional micro-batches are supplied by the CLI.
The authors' source files remain unchanged.
"""
import ast
import json
import math
import os
from pathlib import Path

S1_NOTEBOOK = 'Code/scripts/UMSS/UMSS-S1.ipynb'


def _s1_cells(repository):
    """Read the official training cells, excluding IPython, GPU probes and prediction cells."""
    path = Path(repository) / S1_NOTEBOOK
    cells = json.loads(path.read_text())['cells']
    return path, {index: ast.parse(''.join(cells[index]['source']))
                  for index in (6, 7, 9, 10, 11, 14)}


def s1_defaults(repository):
    """Use the actual notebook's literal recipe rather than maintaining a copy."""
    _, cells = _s1_cells(repository)
    names = {'dataset', 'times', 'load_weights', 'batch_size', 'init_lr',
             'lr_reduce_by', 'reduce_at_epoch', 'n_epochs', 'losses',
             'model_name', 'model_inp_size', 'model_out_size'}
    values = {}
    for index in (6, 7, 9):
        for node in cells[index].body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    if set(values) != names:
        raise ValueError('Incomplete official S1 configuration')
    return values


def run_s1(repository, bindings, wrap_training_generator=None):
    """Execute original S1 data, model, generator, callback and fitting statements.

    Bindings supply the original dependencies, local paths and explicit runtime
    overrides. InteractivePlot is supplied as CSVLogger by our CLI. The only AST
    edit makes the original fit call's verbosity configurable.
    """
    path, cells = _s1_cells(repository)
    state = {**s1_defaults(repository), 'os': os, 'math': math, **bindings}

    def execute(index, nodes=None):
        tree = cells[index] if nodes is None else ast.Module(body=nodes, type_ignores=[])
        exec(compile(tree, f'{path}:cell{index}', 'exec'), state)

    # Keep the notebook's Adam construction and derived timestep count. A
    # micro-batch run explicitly supplies its accumulation optimizer instead.
    derived = [node for node in cells[9].body if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Name)
               and (node.targets[0].id == 'n_timesteps'
                    or (node.targets[0].id == 'opt' and 'opt' not in state))]
    execute(7)
    execute(9, derived)
    for index in (10, 11):
        execute(index)
    if wrap_training_generator is not None:
        state['gen_train'] = wrap_training_generator(state['gen_train'])
    for node in ast.walk(cells[14]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'fit_generator':
            for keyword in node.keywords:
                if keyword.arg == 'verbose':
                    keyword.value = ast.copy_location(ast.Name(id='training_verbose', ctx=ast.Load()), keyword.value)
    ast.fix_missing_locations(cells[14])
    execute(14)
    return state


