"""Fail closed on non-code artifacts and recognizable private literals before upload.

An allowlist scanner assists review; it cannot certify arbitrary text as anonymous.
Run it on the final folder and review any newly added text before publication.
"""
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={'.py','.md','.json','.yaml','.yml','.txt','.toml'}
PATTERNS=[r'[A-Za-z]:[\\/](?:Users|研究|data)[\\/]',
          r'\b(?:WB|AY)_[0-9a-f]{12,}\b',
          r'\b\d{17}[0-9Xx]\b',r'\b1[3-9]\d{9}\b',
          r'[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}',
          r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b']

def main():
    problems=[];count=0
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '.git' in p.parts:continue
        rel=p.relative_to(ROOT)
        if p.name!='.gitignore' and p.suffix.lower() not in ALLOWED:
            problems.append(f'{rel}: non-code/disallowed file type');continue
        try:s=p.read_text(encoding='utf-8')
        except UnicodeError:problems.append(f'{rel}: binary content');continue
        for pattern in PATTERNS:
            if re.search(pattern,s,re.I):problems.append(f'{rel}: potential private literal; review locally')
        count+=1
    if problems:
        print('\n'.join(problems));return 1
    print(f'PASS: {count} text/code files; no bundled data, predictions, models, images or recognized private literals.');return 0

if __name__=='__main__':sys.exit(main())
