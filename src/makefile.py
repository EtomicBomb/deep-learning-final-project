#!/usr/bin/env python

from pathlib import Path

extract_targets = []
recipies = []

for x in Path('data', 'listing.txt').read_text().splitlines():
    init_path = Path('data', 'init', x)
    csv_path = Path('data', 'positions', x).with_suffix('.npz')
    extract_path = Path('data', 'extract', x).with_suffix('.mp4')
    
    extract_targets.append(str(extract_path))
    recipies.append(
        f'{csv_path}: {init_path}\n' 
        f'\tsrc/positions.py {init_path} $@\n'
    )
    recipies.append(
        f'{extract_path}: {csv_path} {init_path}\n' 
        f'\tsrc/extract.py {init_path} {csv_path} $@.part\n'
        f'\tmv $@.part $@\n'
    )

recipies = '\n'.join(recipies)
extract_targets = ' '.join(extract_targets)
Path('Makefile').write_text(f'data: {extract_targets}\n\n{recipies}')
