import math

class Calculator:
    def __init__(self):
        """
        Initialize the calculator with default values.
        Sets result and complex_result to their initial values.
        """
        self.result = 0
        self.complex_result = complex(0, 0)

    def add(self, number):
        """
        Add a number to the result or complex_result.
        If the number is complex, add it to complex_result.
        Otherwise, add it to result.
        """
        if isinstance(number, complex):
            self.complex_result += number
        else:
            self.result += number
        return self.result

    def subtract(self, number):
        """
        Subtract a number from the result or complex_result.
        If the number is complex, subtract it from complex_result.
        Otherwise, subtract it from result.
        """
        if isinstance(number, complex):
            self.complex_result -= number
        else:
            self.result -= number
        return self.result

    def multiply(self, number):
        """
        Multiply the result or complex_result by a number.
        If the number is complex, multiply it with complex_result.
        Otherwise, multiply it with result.
        """
        if isinstance(number, complex):
            self.complex_result *= number
        else:
            self.result *= number
        return self.result

    def divide(self, number):
        """
        Divide the result or complex_result by a number.
        If the number is complex, divide complex_result by it.
        Otherwise, divide result by it.
        Handles division by zero error.
        """
        if isinstance(number, complex):
            if number != 0:
                self.complex_result /= number
            else:
                print("Error: Division by zero.")
        else:
            if number != 0:
                self.result /= number
            else:
                print("Error: Division by zero.")
        return self.result

    def nth_root(self, n):
        """
        Calculate the nth root of the result.
        Handles the case where even roots of negative numbers are not real.
        """
        if self.result < 0 and n % 2 == 0:
            print("Error: Even root of a negative number is not real.")
        else:
            self.result = self.result ** (1/n)
        return self.result

    def mod(self):
        """
        Calculate the modulus (absolute value) of the result or complex_result.
        Returns the absolute value of result or complex_result.
        """
        if isinstance(self.complex_result, complex):
            return abs(self.complex_result)
        else:
            return abs(self.result)

    def reset(self):
        """
        Reset the result and complex_result to their initial values.
        """
        self.result = 0
        self.complex_result = complex(0, 0)
        return self.result

def main():
    """
    Main function to interact with the calculator.
    Continuously prompts the user for an operation and performs the corresponding calculation.
    Handles invalid operations and exits the loop when the user inputs 'exit'.
    """
    calc = Calculator()  # Create an instance of Calculator
    while True:
        # Print the current result and prompt the user for an operation
        print("\nCurrent result: ", calc.result)
        operation = input("Choose an operation (add, subtract, multiply, divide, nth_root, mod, reset, exit): ").strip().lower()

        if operation == "exit":
            break
        elif operation in ["add", "subtract", "multiply", "divide"]:
            # Check if the input includes 'i' to identify complex numbers
            number = complex(input("Enter a number: ")) if 'i' in input else float(input("Enter a number: "))
            if operation == "add":
                calc.add(number)
            elif operation == "subtract":
                calc.subtract(number)
            elif operation == "multiply":
                calc.multiply(number)
            elif operation == "divide":
                calc.divide(number)
        elif operation == "nth_root":
            # Get the value of n for nth root calculation
            n = float(input("Enter the value of n: "))
            calc.nth_root(n)
        elif operation == "mod":
            # Calculate and print the modulus
            print("Modulus: ", calc.mod())
        elif operation == "reset":
            # Reset the calculator
            calc.reset()
        else:
            print("Invalid operation. Please try again.")

if __name__ == "__main__":
    main()
