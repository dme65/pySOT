export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp
	g++ -o sphere_ext_files sphere_ext_files.cpp

.PHONY: test
test: sphere_ext sphere_ext_files
	python test_penalty.py
	python test_projection.py
	python test_ensemble.py
	python test_mixed_integer_constraints.py
	python test_multisearch.py
	python test_simple.py
	python test_subprocess.py
	python test_subprocess_files.py

clean:
	rm sphere_ext
	rm sphere_ext_files
	rm -r ./logfiles/*
	rmdir logfiles