data: data/final/.touch
data/final/.touch: data/cropped/.touch
data/cropped/.touch: data/eye-positions/.touch data/train-test/.touch
data/eye-positions/.touch: data/train-test/.touch
data/train-test/.touch: data/init/.touch

%.touch:
	src/data.py $@
	touch $@

