class HyperOptError(Exception):
    """Exception raised when a field is missing in the input dictionary"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


class PredictOptError(Exception):
    """Exception raised when a field is missing in the input dictionary"""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message