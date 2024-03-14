#!/usr/bin/env python

import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve().parent


print(path)
