export PYTHONPATH=../..:

sphere_ext: sphere_ext.cpp
	g++ -o $@ $<

.PHONY: test
test: sphere_ext
	python test_constraints.py
	python test_ensemble.py
	python test_mixed_integer_constraints.py
	python test_multisearch.py
	python test_simple.py
	python test_subprocess.py