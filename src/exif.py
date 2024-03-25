#!/usr/bin/env python

from pathlib import Path
from subprocess import check_output
import json

out = {}

for source in Path('data', 'listing.txt').read_text().splitlines():
    command = ['exiftool', '-T', '-Rotation', Path('data', 'init', source)]
    out[source] = int(check_output(command).decode('utf-8').strip())

Path('data', 'rotation.json').write_text(json.dumps(out))
