.PHONY: test testmpi clean
export PYTHONPATH=../..:

all:
	g++ -o sphere_ext sphere_ext.cpp -std=c++0x
	g++ -o sphere_ext_files sphere_ext_files.cpp -std=c++0x
	g++ -o sumfun_ext sumfun_ext.cpp -std=c++0x

test: sphere_ext sphere_ext_files sumfun_ext
	for f in test*.py; do python $$f || break 0; done

testmpi: sphere_ext
	for f in mpitest_*.py; do mpiexec -n 5 python $$f || break 0; done

clean:
	rm -f sphere_ext
	rm -f sphere_ext_files
	rm -f sumfun_ext
	rm -rf ./logfiles
