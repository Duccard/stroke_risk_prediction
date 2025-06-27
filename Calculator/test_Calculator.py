import math
from Calculator import Calculator

# Define a test function for the Calculator
def test_calculator():
    calculator = Calculator()
    
    # Test 1: Add two numbers
    calculator.memory = 5  
    calculator.history = ['5']
    calculator.addition(3)
    assert calculator.memory == 8, f"Test failed: Expected 8, got {calculator.memory}"
    print("Test 1 passed: 5 + 3 = 8")
    
    # Test 2: Subtract a number
    calculator.subtraction(2)
    assert calculator.memory == 6, f"Test failed: Expected 6, got {calculator.memory}"
    print("Test 2 passed: 8 - 2 = 6")
    
    # Test 3: Multiply by a number
    calculator.multiply(4)
    assert calculator.memory == 24, f"Test failed: Expected 24, got {calculator.memory}"
    print("Test 3 passed: 6 * 4 = 24")
    
    # Test 4: Divide by a number
    calculator.divide(6)
    assert calculator.memory == 4, f"Test failed: Expected 4, got {calculator.memory}"
    print("Test 4 passed: 24 / 6 = 4")
    
    # Test 5: Square root of a number
    calculator.memory = 16  # Set memory to 16 for square root
    calculator.square_root()
    assert calculator.memory == 4, f"Test failed: Expected 4, got {calculator.memory}"
    print("Test 5 passed: √16 = 4")
    
    # Test 6: Divide by zero (expect error)
    try:
        calculator.divide(0)
    except ValueError as e:
        assert str(e) == "Cannot divide by zero!", f"Test failed: Expected 'Cannot divide by zero!', got {str(e)}"
        print("Test 6 passed: Division by zero raised ValueError")
    
    # Test 7: Square root of a negative number (expect error)
    calculator.memory = -9  # Set memory to -9 for square root
    try:
        calculator.square_root()
    except ValueError as e:
        assert str(e) == "Cannot take the square root of a negative number!", f"Test failed: Expected 'Cannot take the square root of a negative number!', got {str(e)}"
        print("Test 7 passed: Square root of negative number raised ValueError")

    # Test 8: Reset functionality
    calculator.reset()
    assert calculator.memory == 0 and calculator.history == [], f"Test failed: After reset, expected memory 0 and empty history, got memory {calculator.memory}, history {calculator.history}"
    print("Test 8 passed: Reset works correctly")

# Run the test
if __name__ == "__main__":
    test_calculator()
