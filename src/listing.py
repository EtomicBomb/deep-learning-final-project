#!/usr/bin/env python

import os
from pathlib import Path

init = Path('data', 'init')
x = os.walk(init)
x = set(str(Path(name, file).relative_to(init)) for name, _, files in x if len(files) > 0 for file in files)
x = '\n'.join(sorted(x))
Path('data', 'listing.txt').write_text(x)
