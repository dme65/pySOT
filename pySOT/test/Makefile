export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp -std=c++0x
	g++ -o sphere_ext_files sphere_ext_files.cpp -std=c++0x
	g++ -o sumfun_ext sumfun_ext.cpp -std=c++0x

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
	mpiexec -n 5 python test_simple_mpi.py
	mpiexec -n 5 python test_subprocess_mpi.py

clean:
	rm -f sphere_ext
	rm -f sphere_ext_files
	rm -f sumfun_ext
	rm -rf ./logfiles