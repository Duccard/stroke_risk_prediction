# 1) Function to check if the given coordinate is valid on a chessboard
def is_valid_coordinate(coordinate):
    letters = "abcdefgh"
    numbers = "12345678"
    return (
        len(coordinate) == 2 and coordinate[0] in letters and coordinate[1] in numbers
    )


# 2) Function to add a piece to the dictionary if the coordinate is valid and not already occupied
def add_piece(pieces, piece, coordinate):
    if is_valid_coordinate(coordinate):
        if (
            coordinate not in pieces.values()
        ):  # Check if the coordinate is already occupied
            pieces[piece] = coordinate
            print(f"{piece.capitalize()} added at {coordinate}.")
        else:
            print(f"Position {coordinate} is already occupied. Try again.")
    else:
        print(f"Invalid position: {coordinate}. Try again.")


# 3) Function to prompt the user to input a white piece and its position
def get_white_piece(existing_coordinates):
    print("You can choose between the following pieces for the white piece: pawn, king")
    while True:
        white_piece_input = (
            input("Enter the white piece and its position (e.g., 'king a3'): ")
            .lower()
            .strip()
        )
        try:
            piece, coordinate = white_piece_input.split()
            if piece in ["pawn", "king"] and is_valid_coordinate(coordinate):
                if (
                    coordinate not in existing_coordinates
                ):  # Ensure no duplicate position
                    print(f"{piece.capitalize()} at {coordinate} is valid. Continue.")
                    return piece, coordinate
                else:
                    print(
                        f"Position {coordinate} is already occupied by another piece. Try again."
                    )
            else:
                print("Invalid piece or coordinate. Try again.")
        except ValueError:
            print("Invalid format. Use 'piece coordinates' (e.g., 'king a3').")


# 4) Function to prompt the user to input black pieces and their positions
def update_black_pieces(existing_coordinates):
    black_pieces = {}
    print(
        "Enter each black piece and its position (e.g., 'bishop a3'). Type 'done' to finish after adding at least one piece."
    )

    while len(black_pieces) < 16:
        black_piece_input = (
            input("Enter the black piece and its position, or type 'done': ")
            .lower()
            .strip()
        )

        if black_piece_input == "done":
            if len(black_pieces) > 0:
                break
            else:
                print("You need to add at least one black piece.")
                continue

        try:
            piece, coordinate = black_piece_input.split()
            if piece in [
                "pawn",
                "rook",
                "bishop",
                "queen",
                "knight",
                "king",
            ] and is_valid_coordinate(coordinate):
                if (
                    coordinate not in existing_coordinates
                    and coordinate not in black_pieces.values()
                ):  # Check for duplicates
                    add_piece(black_pieces, piece, coordinate)
                    existing_coordinates.add(
                        coordinate
                    )  # Add to the set of occupied coordinates
                else:
                    print(f"Position {coordinate} is already occupied. Try again.")
            else:
                print("Invalid piece or coordinate. Try again.")
        except ValueError:
            print("Invalid format. Use 'piece coordinates' (e.g., 'bishop a3').")

    return black_pieces


# 5) Function to determine which black pieces the white piece can capture
def can_capture(white_piece, white_coordinate, black_pieces):
    captured_pieces = []

    # If the white piece is a king
    if white_piece == "king":
        # Separate the column (file) and row (rank) of the white king's position
        white_column = white_coordinate[0]  # Column (e.g., 'd')
        white_row = int(white_coordinate[1])  # Row as an integer for calculations

        # Loop through each black piece to see if it can be captured
        for black_piece, black_position in black_pieces.items():
            black_column = black_position[0]
            black_row = int(black_position[1])

            # A king captures pieces in any of the 8 adjacent squares
            if (
                abs(ord(black_column) - ord(white_column)) <= 1
                and abs(black_row - white_row) <= 1
            ):
                captured_pieces.append((black_piece, black_position))

    # If the white piece is a pawn
    elif white_piece == "pawn":
        # Separate the column (file) and row (rank) of the white pawn's position
        white_column = white_coordinate[0]  # Column (e.g., 'd')
        white_row = int(white_coordinate[1])  # Row as an integer for calculations

        # Loop through each black piece to see if it can be captured
        for black_piece, black_position in black_pieces.items():
            black_column = black_position[0]
            black_row = int(black_position[1])

            # Pawns capture diagonally one square forward
            # So we check if the black piece is in one of those diagonal squares
            if (
                black_row == white_row + 1
                and abs(ord(black_column) - ord(white_column)) == 1
            ):
                captured_pieces.append((black_piece, black_position))

    # Return the list of captured black pieces
    return captured_pieces


# 6) Main function that integrates all steps and handles the program logic
def main():
    existing_coordinates = set()  # Set to store all occupied coordinates

    # 6.1) Get the white piece and its position
    white_piece, white_coordinate = get_white_piece(existing_coordinates)
    existing_coordinates.add(
        white_coordinate
    )  # Add white piece's position to occupied coordinates

    # 6.2) Get the black pieces and their positions
    black_pieces = update_black_pieces(existing_coordinates)
    print("Added black pieces:")
    for piece, coordinate in black_pieces.items():
        print(f"{piece.capitalize()} at {coordinate}")

    # 6.3) Determine which black pieces the white piece can capture
    captures = can_capture(white_piece, white_coordinate, black_pieces)
    if captures:
        print("Black pieces the white piece can capture:")
        for piece, coordinate in captures:
            print(f"{piece.capitalize()} at {coordinate}")
    else:
        print("No black pieces can be captured by the white piece.")


# 7) Ensure the main function is called only when the script is executed directly
if __name__ == "__main__":
    main()
