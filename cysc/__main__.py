"""python -m cysc --config <local.yaml> --stage all"""
import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import argparse
import sys
import importlib.metadata
from pathlib import Path
from .io import load_config,output,read_json,write_json,digest,ROOT

STAGES=['prepare','development','reduction','stability','final','external','shap','targeted','profiles','figures','tables']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['all']+STAGES,required=True)
    args=parser.parse_args();cfg=load_config(args.config)
    from . import preprocess,models,evaluation,figures,tables
    functions={'prepare':preprocess.run,'development':models.development,'reduction':models.reduction,'stability':models.stability,
               'final':models.final,'external':evaluation.external,'shap':models.global_shap,'targeted':evaluation.targeted,
               'profiles':evaluation.profiles,'figures':figures.run,'tables':tables.run}
    cfgfile=Path(args.config)
    stamp=output(cfg)/'execution_contract.json'
    current={'config_sha256':digest(cfgfile),'frozen_sha256':digest(ROOT/'config/frozen_analysis.json'),
             'code_sha256':{str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'cysc').glob('*.py'))}}
    if stamp.exists() and read_json(stamp)!=current:raise ValueError('Configuration/code changed during this run; use a new output folder.')
    if not stamp.exists():write_json(stamp,current)
    write_json(output(cfg)/'runtime_environment.json',{'python':sys.version.split()[0],
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','xgboost','lightgbm','shap','statsmodels','scipy','matplotlib']}})
    stages=STAGES if args.stage=='all' else [args.stage]
    for stage in stages:
        state=output(cfg)/'state.json'
        if state.exists() and stage in read_json(state)['completed']:raise ValueError('Stage already completed; do not overwrite.')
        if state.exists() and 'external' in read_json(state)['completed'] and stage in ['prepare','development','reduction','stability','final']:
            raise ValueError('Redevelopment after external evaluation is prohibited.')
        print('Starting',stage,flush=True);functions[stage](cfg)
    print('Finished. Generated files are private local outputs; do not upload them with the code repository.',flush=True)

if __name__=='__main__':main()
