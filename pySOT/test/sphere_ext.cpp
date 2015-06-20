#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <numeric>
#include <random>

int main(int argc, char** argv) {
	// Random number generator
	std::random_device                  rand_dev;
	std::mt19937                        generator(rand_dev());
	std::uniform_real_distribution<float>  distr(0.0, 1.0);

	// Pretend the simulation crashes with probability 0.1
  	if(distr(generator) > 0.1) {

		// Convert input to a standard vector
		std::vector<float> vect;
		std::stringstream ss(argv[1]);
		float f;

		while (ss >> f) {
    		vect.push_back(f);
			if (ss.peek() == ',')
    	    	ss.ignore();
		}
   	 	sleep(1);    
   	 	printf("%g\n", std::inner_product( vect.begin(), vect.end(), vect.begin(), 0.0 ));
	}
    return 0;
}