from pathlib import Path
from utilities.color import BLACK, WHITE

#### Directories ####
MANIM_DIR = Path(__file__).parent.resolve() #the absloute path towards manim
MEDIA_DIR = MANIM_DIR/ "media"
VIDEO_DIR = MANIM_DIR/ "video"
IMAGE_DIR = MANIM_DIR/ "image"


### Default redering settings ###

# Full HD resolution (1920,1080)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
# Frames per second
FRAME_RATE = 30 

DEFAULT_STROKE_WIDTH = 4
DEFAULT_BACKGROUND_COLOR = BLACK
DEFAULT_STROKE_COLOR = BLACK
DEFAULT_FILL_COLOR = WHITE
DEFAULT_FILL_OPACITY = 1.0

### Rendering options ###
ENABLE_ANTIALIASING = True # makes curves smoother by partially coloring some pixels
PIXEL_SCALE = 1.0 # 1.0 for default resolution (1 pixel per frame-unit) 2.0 for high resolution, etc ...

### Animation system ###
DEFAULT_ANIMATION_DURATION = 1.0  # in seconds
EASING_FUNCTION = "linear"        # or "ease_in_out", etc.

### Logging and debug ###
VERBOSE = True # for an interactive experience (it is a flag that tells us when to show a message in the terminal) 
DEBUG_MODE = False                              

# example where VERBOSE = True
""" if VERBOSE:
         print("Generating bezier path...") """