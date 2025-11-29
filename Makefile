PYTEST := RLVR_SMOKE=1 conda run -n torch312 pytest

.PHONY: test
test:
	$(PYTEST)
