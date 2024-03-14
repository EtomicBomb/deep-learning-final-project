#!/usr/bin/env python

import sys
from pathlib import Path

destination = Path(sys.argv[1]).parent

sources = [Path(source).parent for source in sys.argv[2:]]

print(destination, sources)
