export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp -std=c++0x
	g++ -o sphere_ext_files sphere_ext_files.cpp -std=c++0x
	g++ -o sumfun_ext sumfun_ext.cpp -std=c++0x

.PHONY: test
test: sphere_ext sphere_ext_files sumfun_ext
	python3 test_simple.py
	python3 test_rbf.py
	python3 test_penalty.py
	python3 test_projection.py
	python3 test_ensemble.py
	python3 test_mars.py
	python3 test_mixed_integer_constraints.py
	python3 test_multisampling.py
	python3 test_subprocess.py
	python3 test_subprocess_files.py
	python3 test_subprocess_partial_info.py
	mpiexec -n 5 python3 test_simple_mpi.py
	mpiexec -n 5 python3 test_subprocess_mpi.py

clean:
	rm -f sphere_ext
	rm -f sphere_ext_files
	rm -f sumfun_ext
	rm -rf ./logfiles
