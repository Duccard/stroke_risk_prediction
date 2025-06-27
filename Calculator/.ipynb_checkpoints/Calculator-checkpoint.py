import math
from typing import List

class Calculator:
    """Performs basic arithmetic operations and stores the result."""

    def __init__(self) -> None:
        """Initializes the calculator with memory and history."""
        # Stores the current result
        self.memory: float = 0.0 
        # Stores input history (numbers, operations)
        self.history: List[str] = []  

    def addition(self, number: float) -> None:
        """Performs addition and stores the result in memory."""
        self.memory += number

    def subtraction(self, number: float) -> None:
        """Performs subtraction and stores the result in memory."""
        self.memory -= number

    def multiply(self, number: float) -> None:
        """Performs multiplication and stores the result in memory."""
        self.memory *= number

    def divide(self, number: float) -> None:
        """Performs division and stores the result in memory."""
        if number == 0:
            raise ValueError("Cannot divide by zero!")
        self.memory /= number

    def square_root(self) -> None:
        """Calculates the square root of the current memory value."""
        if self.memory < 0:
            raise ValueError("Cannot take the square root of a negative number!")
        self.memory = math.sqrt(self.memory)

    def reset(self) -> None:
        """Resets the calculator's memory and history."""
        self.memory = 0.0
        self.history = []

    def show_progress(self, include_result: bool = True) -> None:
        """Displays the history of operations and optionally the result."""
        progress: str = " ".join(self.history)
        if include_result:
            print(f"Progress: {progress} = {self.memory}")
        else:
            print(f"Progress: {progress}")


def main() -> None:
    """Main function to perform calculator operations based on user input. Prompts user for an operation symbol and performs the corresponding calculation."""
    calculator: Calculator = Calculator()
    print("Welcome to Calculator! Type 'c' at any stage to clear memory")
    print(
        "Note: If you want to calculate the root of a number, put the root symbol after the number is inserted. "
    )
    while True:
        user_input: str = input("Enter number or 'c': ").strip()
        if user_input.lower() == "c":
            calculator.reset()
            print("Memory reset.")
            continue

        try:
            number: float = float(user_input)
            calculator.memory = number
            # Start history with the first number
            calculator.history = [f"{number}"]
            # Show progress without the result
            calculator.show_progress(include_result=False)
        except ValueError:
            print("Invalid input. Please enter a number or 'c'.")
            continue

        while True:
            symbol: str = input(
                "Choose one of the following symbols: +, -, *, /, √ "
            ).strip()
            if symbol.upper() == "C":
                calculator.reset()
                print("Memory reset.")
                break
            if symbol not in ["+", "-", "*", "/", "√"]:
                print(
                    "Invalid symbol. Please choose one of the allowed symbols: +, -, *, /, √"
                )
                continue

            # Append the operator symbol to the history
            calculator.history.append(symbol)
            # Show progress without result
            calculator.show_progress(include_result=False)

            if symbol == "√":
                try:
                    calculator.square_root()
                    # Append to history after calculation
                    calculator.history.append("√")
                except ValueError as e:
                    print(e)
                    break
            else:
                try:
                    number2: float = float(input("Enter a number: "))
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

            # After performing the calculation, show the progress including the result
            calculator.show_progress(include_result=True)

            # Print the final result
            print(f"Result: {calculator.memory}")

            # Ask the user if they want to continue or reset after showing the result
            continue_choice: str = (
                input(
                    "Do you want to continue with another operation on this result? ('y' to continue, 'c' to reset): "
                )
                .strip()
                .lower()
            )
            if continue_choice == "c":
                calculator.reset()
                print("Memory reset.")
                break
            elif (
                continue_choice != "y"
            ):
                calculator.reset()
                print("Memory reset.")
                break
            # Clear the history and set it to only the latest result
            else:
                calculator.history = [f"{calculator.memory}"]
                calculator.show_progress(include_result=False)


if __name__ == "__main__":
    """Ensures the main function runs only if the script is executed directly."""
    main()
