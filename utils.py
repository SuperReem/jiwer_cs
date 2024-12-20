import re
from tabulate import tabulate

def is_arabic(text):
    """
    Checks if a given text is classified as Arabic.

    Args:
        word (str): The input text.

    Returns:
        bool: True if the text is classified as Arabic, False otherwise.
    """
    return re.search(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+', text)



def display_metrics_table(metrics, title, headers=None):
    """
    Prints the given metrics in a formatted table.

    Args:
    metrics (list of tuples): List of tuples containing metric names and their values.
    headers (list of str, optional): List of column headers. If None, default headers will be used.
    """
    if headers is None:
        headers = ["Metric", "Value"]  # Default headers
        
    print(f"### {title} ###")
    print(tabulate(metrics, headers=headers, tablefmt="pretty"))
    print()
