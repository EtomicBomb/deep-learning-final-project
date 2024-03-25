#!/usr/bin/env python

from pathlib import Path
from math import ceil
import json
from random import shuffle

participants = Path('data', 'listing.txt').read_text().splitlines()
participants = list(set(str(Path(p).parent) for p in participants))
shuffle(participants)

train_count = ceil(len(participants) * 0.8)
test_count = ceil(len(participants) * 0.1)

train = participants[:train_count]
test = participants[train_count:train_count + test_count]
validation = participants[train_count + test_count:]

assert len(train) > 0
assert len(test) > 0
assert len(validation) > 0

Path('data', 'split.json').write_text(json.dumps(dict(
    train=train,
    test=test,
    validation=validation,
)))


