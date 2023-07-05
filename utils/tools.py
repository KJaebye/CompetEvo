def str2bool(input_str):
    """Converts a string to a boolean value.

    Args:
        input_str (str): The string to be converted.

    Returns:
        bool: The boolean representation of the input string.
    """
    true_values = ["true", "yes", "1", "on", "y", "t"]
    false_values = ["false", "no", "0", "off", "n", "f"]

    lowercase_str = input_str.lower()
    if lowercase_str in true_values:
        return True
    elif lowercase_str in false_values:
        return False
    else:
        raise ValueError("Invalid input string. Could not convert to boolean.")