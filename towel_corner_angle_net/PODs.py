from typing import Tuple, Union





class Coord:
    __slots__ = ["x", "y"]

    def __init__(self, x: Union[int, float], y: Union[int, float]):
        assert type(x) == type(y)

        self.x: Union[int, float] = x
        self.y: Union[int, float] = y

class SampleMetaData:
    """
    A class to represent metadata associated with a sample.

    Attributes:
        path (str): The file path or location of the sample.
        center (Tuple[int, int]): The coordinates (x, y) representing the center of the sample.
        angle (float): The angle of the sample relative to a reference, typically in degrees.

    Methods:
        __init__(self, path: str, center: Tuple[int, int], angle: float): Initializes the SampleMetaData object.
    """

    # Declaring slots to restrict attribute creation to the ones listed here. This optimizes memory usage.
    __slots__ = ["path", "center", "angle"]

    def __init__(self, path: str, center: Coord, angle: float):
        """
        Constructs all the necessary attributes for the SampleMetaData object.

        Parameters:
            path (str): The file path or location of the sample.
            center (Tuple[int, int]): The coordinates (x, y) representing the center of the sample.
            angle (float): The angle of the sample relative to a reference, typically in degrees.
        """
        self.path: str = path
        self.center: Coord = center
        self.angle: float = angle