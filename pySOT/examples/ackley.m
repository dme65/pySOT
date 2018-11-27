function val = ackley(x)
    dim = length(x);
    val = -20*exp(-0.2*sqrt(sum(x.^2,2)/dim)) - ...
        exp(sum(cos(2*pi*x),2)/dim) + 20 + exp(1);