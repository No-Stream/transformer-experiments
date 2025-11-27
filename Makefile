PYTEST := conda run -n torch312 pytest

.PHONY: test
test:
	$(PYTEST)
