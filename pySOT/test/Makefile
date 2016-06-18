export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp
	g++ -o sphere_ext_files sphere_ext_files.cpp
	g++ -o sumfun_ext sumfun_ext.cpp

.PHONY: test
test: sphere_ext sphere_ext_files sumfun_ext
	python test_simple.py
	python test_penalty.py
	python test_projection.py
	python test_ensemble.py
	python test_mixed_integer_constraints.py
	python test_multisampling.py
	python test_subprocess.py
	python test_subprocess_files.py
	python test_subprocess_partial_info.py
	python test_matlab_engine.py

clean:
	rm sphere_ext
	rm sphere_ext_files
	rm sumfun_ext
	rm -r ./logfiles/*
	rmdir logfiles