#include <iostream>
#include <vector>
#include <sstream>
#include <unistd.h>
#include <numeric>
#include <random>
#include <chrono>
#include <thread>

/*
Simple routine that takes an input of the form 'x1,x2,...,xn' and converts this
input to a standard vector of floats. The range of objective function values is
from 0 to 436.6.
*/

int main(int argc, char** argv) {

    // Convert input to a standard vector
    std::vector<float> vect;
    std::stringstream ss(argv[1]);
    float f;

    while (ss >> f) {
        vect.push_back(f);
        if (ss.peek() == ',')
            ss.ignore();
    }

    double ssum = 0.0;
    double prod = 1.0;
    for(int i=0; i<vect.size(); i++) {
        prod *= vect[i];
    }

    for(int i=1; i < 1000; i++) {
        ssum += (float(i)/1000) * std::abs(sin(prod*(float(i)/1000)));
        if (i % 100 == 0) {
            printf("%g\n", ssum);
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Sleep for 0.1 seconds
        }
    }

    printf("%g\n", ssum);
    return 0;
}