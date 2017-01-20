# coding=utf-8

import numpy as np
import random


# First implement a gradient checker by filling in the following functions
def gradcheck_naive(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x)  # Evaluate function value at original point
    h = 1e-4    # h即delta_x

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### try modifying x[ix] with h defined above to compute numerical gradients
        ### make sure you call random.setstate(rndstate) before calling f(x) each time, this will make it 
        ### possible to test cost functions with built in randomness later

        # 利用导数的定义求导
        old_value = x[ix]
        x[ix] = old_value - h
        random.setstate(rndstate)
        f1 = f(x)[0]
        x[ix] = old_value + h
        random.setstate(rndstate)
        f2 = f(x)[0]
        numgrad = (f2 - f1) / (2.0 * h)
        x[ix] = old_value

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return

        it.iternext()  # Step to next dimension

    print "Gradient check passed!"


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print "Running sanity checks..."
    gradcheck_naive(quad, np.array(123.456))  # scalar test
    gradcheck_naive(quad, np.random.randn(3, ))  # 1-D test
    gradcheck_naive(quad, np.random.randn(4, 5))  # 2-D test
    print ""


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_gradcheck.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    from q2_sigmoid import sigmoid, sigmoid_grad
    f = lambda x: (sigmoid(x), sigmoid_grad(sigmoid(x)))
    gradcheck_naive(f, np.arange(6))


if __name__ == "__main__":
    sanity_check()
