#!/usr/bin/env python

from pathlib import Path
import subprocess

extract_targets = []
recipies = []

for x in Path('data', 'listing.txt').read_text().splitlines():
    csv_path = Path('data', 'positions', x).with_suffix('.npz')
    init_path = Path('data', 'init', x)
    subprocess.run([
        'sbatch', 
        'src/batch.sh', 
        csv_path,
    ])
