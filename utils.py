import re

def is_arabic(text):
    """
    Checks if a given text is classified as Arabic.

    Args:
        word (str): The input text.

    Returns:
        bool: True if the text is classified as Arabic, False otherwise.
    """
    return re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', text)
