.PHONY: example mpiexample clean
export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp -std=c++0x
	g++ -o sphere_ext_files sphere_ext_files.cpp -std=c++0x
	g++ -o sumfun_ext sumfun_ext.cpp -std=c++0x

example: sphere_ext sphere_ext_files sumfun_ext
	for f in example*.py; do python3 $$f || break 0; done

mpiexample: sphere_ext
	for f in mpiexample_*.py; do mpiexec -n 5 python3 $$f || break 0; done

clean:
	rm -f sphere_ext
	rm -f sphere_ext_files
	rm -f sumfun_ext
	rm -rf ./logfiles
