from colour import Color

# Standard colors (all colors aren't equal)
WHITE = Color("white")
BLACK = Color("black")
RED = Color("red")
GREEN = Color("green")
BLUE = Color("blue")
YELLOW = Color("Yellow")
PURPLE = Color("purple") # Hollow purple go brrrr
ORANGE = Color("orange")

def ensure_color(color):
    if isinstance(color, Color):
        return color # just beware from the border effects(we're not using copies here)

    if isinstance(color, str):
        try:
            return Color(color.lower())
        except Exception:
            raise ValueError("Invalid color string")

    if isinstance(color, tuple) and len(color) == 3:

        if all(0 <= c <= 1 for c in color):
            return Color(rgb=color)
        elif all(0 <= c <= 255 for c in color):
            return Color(rgb=tuple(c / 255 for c in color))
        else:
            raise ValueError("RGB tuple values must be in range [0, 1] or [0, 255]")
    
    raise TypeError(f"Cannot convert type {type(color)} to Color")


def ensure_rgb(color):
    """
    Converts Color, string, or tuple to RGB tuple (r,g,b) with floats 0..1
    """
    c = ensure_color(color)  # your current function returns a Color object
    return c.get_rgb()       # tuple (r,g,b)


def hex_to_rgb(hex_color):
    """
    Converts a hex string "#RRGGBB" (a byte for each color) to RGB tuples
    """
    hex = ensure_color(hex_color)
    return hex.get_rgb()

def rgb_to_hex(rgb_color):
    """
    Converts an RGB tuple (values between 0 and 1) to a hex string "#RRGGBB"
    """

    rgb = ensure_color(rgb_color)
    r, g, b = rgb.get_rgb()
    #return rgb.get_hex()
    return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))

def interpolate_colors(color1, color2, t): # in every single function here we suppose that arguments are Color object
    # extract colors
    color1_ensure = ensure_color(color1)
    color2_ensure = ensure_color(color2)

    r1, g1, b1 = color1_ensure.get_rgb()
    r2, g2, b2 = color2_ensure.get_rgb()

    r = (1 - t)*r1 + t*r2
    g = (1 - t)*g1 + t*g2
    b = (1 - t)*b1 + t*b2

    return r, g, b

def lighten(color, amount= 0.1):
    return interpolate_colors(color, WHITE, amount)


def darken(color, amount= 0.1):
    return interpolate_colors(color, BLACK, amount)


def color_gradient(colors, n): 
    gradient = []
    n_colors = len(colors) # n > n_colors

    intervals = n_colors - 1
    if n_colors < 2:
        return [colors]*n
    n_per_interval = n//intervals
    remains = n % intervals

    for i in range(intervals): 
        steps = n_per_interval
        # we know for sure that remains < intervals (euclidian division theorem)
        if i < remains : # in this case we still need to cover the remains
            steps += 1
        for j in range(steps):
            t = j/max(steps - 1, 1) # avoid division by 0
            gradient.append(interpolate_colors(colors[i], colors[i + 1], t))

    return gradient



def blend(colors):
    if not colors:
        raise ValueError("Need at least one color to blend")
    colors_ensure = [ensure_color(color) for color in colors]
    n = len(colors_ensure)
    
    r = sum(color.get_red() for color in colors_ensure)/n
    g = sum(color.get_green() for color in colors_ensure)/n
    b = sum(color.get_blue() for color in colors_ensure)/n

    return Color(rgb = (r, g, b))

def invert_color(color):
    color_e =ensure_color(color)
    r, g, b = color_e.get_rgb()
    r, g, b = 1 - r, 1 - g, 1 - b # In Colors RGB is normalized

    return Color(rgb = (r, g, b)) 

######## Tests ###########

# Conversion test

assert rgb_to_hex((1, 0, 0)) == "#FF0000"
""" 
print(hex_to_rgb("#00FF00"))
print(Color("green").get_rgb())

assert hex_to_rgb("#00FF00") == Color("green").get_rgb()
 """
# Blend test
def colors_almost_equal(c1, c2, tol=1e-6):
    r1, g1, b1 = ensure_color(c1).get_rgb()
    r2, g2, b2 = ensure_color(c2).get_rgb()
    return all(abs(a - b) < tol for a, b in zip((r1, g1, b1), (r2, g2, b2)))

assert colors_almost_equal(blend(["red", "blue"]), Color(rgb = (0.5, 0.0, 0.5)))


# Invert test
assert invert_color("white") == Color("black")
assert invert_color("black") == Color("white")

#gradient test
gradient = color_gradient(["red", "blue"], 3)
print(gradient)
print('#7F007F')
assert len(gradient) == 3
assert colors_almost_equal(gradient[0], "red")
assert colors_almost_equal(gradient[-1], "blue")