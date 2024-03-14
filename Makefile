data: data/final/.touch
data/final/.touch: data/final/build.py data/cropped/.touch
data/cropped/.touch: data/cropped/build.py data/eye-positions/.touch data/train-test/.touch
data/eye-positions/.touch: data/eye-positions/build.py data/train-test/.touch
data/train-test/.touch: data/train-test/build.py data/init/.touch
data/init/.touch: data/init/build.py

%.touch: 
	$(dir $@)build.py $<
	touch $@

