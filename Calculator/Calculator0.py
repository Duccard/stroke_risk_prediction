import math

class Calculations(object):
    def __init__(self, real, imaginary):
        """
        Initialize the Calculations class with real and imaginary parts.
        """
        self._real = real
        self._imaginary = imaginary

    @property
    def real(self):
        """
        Property to get the real part of the complex number.
        """
        return self._real

    @property
    def imaginary(self):
        """
        Property to get the imaginary part of the complex number.
        """
        return self._imaginary

    def __add__(self, no):
        """
        Add two complex numbers.
        """
        return Calculations(self.real + no.real, self.imaginary + no.imaginary)

    def __sub__(self, no):
        """
        Subtract one complex number from another.
        """
        return Calculations(self.real - no.real, self.imaginary - no.imaginary)

    def __mul__(self, no):
        """
        Multiply two complex numbers.
        """
        real = self.real * no.real - self.imaginary * no.imaginary
        imaginary = self.real * no.imaginary + self.imaginary * no.real
        return Calculations(real, imaginary)

    def __truediv__(self, no):
        """
        Divide one complex number by another.
        """
        denominator = no.real ** 2 + no.imaginary ** 2
        real = (self.real * no.real + self.imaginary * no.imaginary) / denominator
        imaginary = (self.imaginary * no.real - self.real * no.imaginary) / denominator
        return Calculations(real, imaginary)

    def mod(self):
        """
        Calculate the modulus of the complex number.
        """
        return Calculations(math.sqrt(self.real ** 2 + self.imaginary ** 2), 0)

    def __str__(self):
        """
        String representation of the complex number.
        """
        if self.imaginary == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.imaginary >= 0:
                result = "0.00+%.2fi" % (self.imaginary)
            else:
                result = "0.00-%.2fi" % (abs(self.imaginary))
        elif self.imaginary > 0:
            result = "%.2f+%.2fi" % (self.real, self.imaginary)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.imaginary))
        return result

if __name__ == '__main__':
    """
    Main function to read input, perform calculations, and print results.
    """
    num_tests = int(input())
    results = []

    for _ in range(num_tests):
        """
        Read the real and imaginary parts for two complex numbers.
        """
        c = map(float, input().split())
        d = map(float, input().split())
        x = Calculations(*c)
        y = Calculations(*d)
        results.append(str(x+y))
        results.append(str(x-y))
        results.append(str(x*y))
        results.append(str(x/y))
        results.append(str(x.mod()))
        results.append(str(y.mod()))

    """
    Print all results for each test case.
    """
    for result in results:
        print(result)
