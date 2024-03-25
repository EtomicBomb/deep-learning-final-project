#!/usr/bin/env python

from pathlib import Path
import subprocess

for x in Path('data/listing.txt').read_text().splitlines():
    subprocess.run([
        'sbatch', 
        'src/batch.sh', 
        Path('data/extract', x).with_suffix('.mp4'),
    ])
