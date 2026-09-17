"""Local-only I/O. Inputs, outputs and trained objects must stay outside this repository."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

FROZEN = read_json(ROOT / 'config/frozen_analysis.json')
DICTIONARY = FROZEN['variable_dictionary']

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              default=lambda x: x.item() if hasattr(x, 'item') else str(x)), encoding='utf-8')

def outside_repo(path):
    p = Path(path).expanduser().resolve()
    if p == ROOT or ROOT in p.parents:
        raise ValueError('Private inputs and generated outputs must be outside the code repository.')
    return p

def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['_base'] = str(path.parent)
    cfg['output_dir'] = str(resolve_path(cfg, cfg['output_dir']))
    if cfg.get('mode') not in ('paper_reproduction', 'synthetic_test'):
        raise ValueError('Use paper_reproduction or synthetic_test; this is not an automatic model-design tool.')
    return cfg

def resolve_path(cfg, path):
    p = Path(path).expanduser()
    return outside_repo(p if p.is_absolute() else Path(cfg['_base']) / p)

def read_table(path):
    path = outside_repo(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() in ('.xlsx', '.xlsm'):
        return pd.read_excel(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError('Supported private inputs: parquet, xlsx, csv.')

def output(cfg):
    p = outside_repo(cfg['output_dir'])
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_table(cfg, name, table):
    p = output(cfg) / 'tables' / (name + '.csv')
    p.parent.mkdir(exist_ok=True)
    table.to_csv(p, index=False)

def save_private(cfg, name, table):
    p = output(cfg) / 'private' / (name + '.parquet')
    p.parent.mkdir(exist_ok=True)
    table.to_parquet(p, index=False)

def load_private(cfg, name):
    return pd.read_parquet(output(cfg) / 'private' / (name + '.parquet'))

def save_arrays(cfg, name, **arrays):
    p = output(cfg) / 'private' / (name + '.npz')
    p.parent.mkdir(exist_ok=True)
    np.savez_compressed(p, **arrays)

def arrays(cfg, name):
    return np.load(output(cfg) / 'private' / (name + '.npz'), allow_pickle=False)

def checkpoint(cfg, stage):
    p = output(cfg) / 'state.json'
    state = read_json(p) if p.exists() else {'completed': []}
    if stage in state['completed']:
        raise ValueError('Stage already completed. Use a fresh output directory instead of overwriting.')
    state['completed'].append(stage)
    state['frozen_spec_sha256'] = digest(ROOT / 'config/frozen_analysis.json')
    write_json(p, state)

def require(cfg, stage):
    state = read_json(output(cfg) / 'state.json')
    if stage not in state['completed']:
        raise ValueError('Required preceding stage: ' + stage)
    if state['frozen_spec_sha256'] != digest(ROOT / 'config/frozen_analysis.json'):
        raise ValueError('Frozen specification changed during execution.')
    receipt=output(cfg)/'cohort_receipt.json'
    if receipt.exists():
        for name,h in read_json(receipt).items():
            if digest(output(cfg)/'private'/(name+'.parquet'))!=h:
                raise ValueError('Prepared cohort changed during this run.')
