# Primary colors
import matplotlib.colors

BLUE = "#1E64C8"
YELLOW = "#FFD200"
WHITE = "#FFFFFF"
BLACK = "#000000"
PRIMARY_COLORS = [BLUE, YELLOW, WHITE, BLACK]

# Secondary colors
ORANGE = "#F1A42B"
RED = "#DC4E28"
AQUA = "#2D8CA8"
PINK = "#E85E71"
SKY = "#8BBEE8"
LIGHTGREEN = "#AEB050"
PURPLE = "#825491"
WARMORANGE = "#FB7E3A"
TURQUOISE = "#27ABAD"
LIGHTPURPLE = "#BE5190"
GREEN = "#71A860"
SECONDARY_COLORS = [ORANGE, RED, AQUA, PINK, SKY, LIGHTGREEN, PURPLE,
                    WARMORANGE, TURQUOISE, LIGHTPURPLE, GREEN]


def bgr(c):
    return rgb(c)[::-1]


def rgb(c):
    return tuple(
        int(channel * 255) for channel in matplotlib.colors.to_rgb(c))


def rgba(c, alpha=0.9):
    c = matplotlib.colors.to_rgba(c)
    c = c[:3] + (alpha,)

    return c
