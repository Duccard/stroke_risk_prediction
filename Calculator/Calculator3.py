import math

class Calculator:
    """Performs basic arithmetic operations and stores the result."""
    
    def __init__(self):
        self.memory = 0
        self.history = []  # Stores input history (numbers, operations)
    
    def addition(self, number):
        self.memory += number

    def subtraction(self, number):
        self.memory -= number

    def multiply(self, number):
        self.memory *= number

    def divide(self, number):
        if number == 0:
            raise ValueError("Cannot divide by zero!")
        self.memory /= number

    def square_root(self):
        if self.memory < 0:
            raise ValueError("Cannot take the square root of a negative number!")
        self.memory = math.sqrt(self.memory)

    def reset(self):
        self.memory = 0
        self.history = []

    def show_progress(self, include_result=True):
        """ Display history up to this point, including the result if requested. """
        progress = " ".join(self.history)
        if include_result:
            print(f"Progress: {progress} = {self.memory}")
        else:
            print(f"Progress: {progress}")

def main():
    """ Main function to perform calculator operations based on user input. Prompts user for an operation symbol and performs the corresponding calculation. """
    calculator = Calculator()
    print("Welcome to Calculator! Type 'c' at any stage to clear memory.")
    
    while True:
        user_input = input("Enter number or 'c': ").strip()
        if user_input.lower() == "c":
            calculator.reset()
            print("Memory reset.")
            continue
        
        try:
            number = float(user_input)
            calculator.memory = number
            calculator.history = [f"{number}"]  # Start history with the first number
            calculator.show_progress(include_result=False)
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")
            continue

        while True:
            symbol = input("Choose one of the following symbols: +, -, *, /, √ ").strip()
            if symbol.upper() == "c":
                calculator.reset()
                print("Memory reset.")
                break
            if symbol not in ["+", "-", "*", "/", "√"]:
                print("Invalid symbol. Please choose one of the allowed symbols: +, -, *, /, √")
                continue

            # Append the operator symbol to the history
            calculator.history.append(symbol)
            calculator.show_progress(include_result=False)

            if symbol == "√":
                try:
                    calculator.square_root()
                    calculator.history.append("√")  # Append to history after calculation
                except ValueError as e:
                    print(e)
                    break
            else:
                try:
                    number2 = float(input("Enter a number: "))
                    # Perform the operation and store the result
                    if symbol == "+":
                        calculator.addition(number2)
                    elif symbol == "-":
                        calculator.subtraction(number2)
                    elif symbol == "*":
                        calculator.multiply(number2)
                    elif symbol == "/":
                        try:
                            calculator.divide(number2)
                        except ValueError as e:
                            print(e)
                            continue
                    
                    # Append the second number after the operation
                    calculator.history.append(f"{number2}")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
                    continue

            print(f"Result: {calculator.memory}")
            calculator.show_progress()

            # Ask the user if they want to continue or reset after showing the result
            continue_choice = input("Do you want to continue with another operation on this result? ('y' to continue, 'c' to reset): ").strip().lower()
            if continue_choice == "c":
                calculator.reset()
                print("Memory reset.")
                break  # Exit to the main loop to start fresh
            elif continue_choice != "y":  # If the user presses anything other than "y" or "C", reset
                calculator.reset()
                print("Memory reset.")
                break  # Exit to the main loop
            else:  # Clear the history and set it to only the latest result
                calculator.history = [f"{calculator.memory}"]
                calculator.show_progress(include_result=False)

if __name__ == "__main__":
    """ Ensures the main function runs only if the script is executed directly. """
    main()
