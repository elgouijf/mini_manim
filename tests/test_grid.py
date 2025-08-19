import mobjects.mobjects as mbj
import manim_bg.renderer as rdr
import cairo
from utilities.color import BLACK, WHITE,RED, GREEN, BLUE

import numpy as np
WIDTH, HEIGHT, FPS = 1920, 1080, 30

# Cairo setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)


# Create renderer with video recording
renderer = rdr.Renderer(surface, ctx, FPS,WIDTH, HEIGHT, file_name="grid.mp4")
grid = mbj.Line.grid(WIDTH, HEIGHT, 20, 20)
for line in grid:
    line.stroke_color = (0, 0, 0)  # Set line color to black
    line.stroke_width = 1  # Set line width
a = float(input("Enter a: "))
b = float(input("Enter b: "))
c = float(input("Enter c: "))
d = float(input("Enter d: "))
# Create a transformation matrix
starts = {line:line.start_pt for line in grid}
ends = {line:line.end_pt for line in grid}
rgb_RED = RED.get_rgb()
Text = mbj.Text("Grid Animation of the matrix",position=(WIDTH/3,HEIGHT/10), color=rgb_RED, font_size=50)

for frame in range(1000):
    # Clear the context
    ctx.set_source_rgba(1, 1, 1, 1)  # White background
    ctx.paint()
    
    # Draw the grid
    
    for line in grid:
        renderer.render_line(line)
        renderer.render_text(Text)
        
        # Apply transformation to each line
        matrix = np.array([[a*frame/1000 + (1-frame/1000)*1, b*frame/1000], [c*frame/1000, d*frame/1000 + (1-frame/1000)*1]])
        transformed_start = np.dot(matrix, starts[line]-np.array([WIDTH/2, HEIGHT/2]))
        transformed_end = np.dot(matrix, ends[line]-np.array([WIDTH/2, HEIGHT/2]))
        line.start_pt = transformed_start + np.array([WIDTH/2, HEIGHT/2])
        line.end_pt = transformed_end+  np.array([WIDTH/2, HEIGHT/2])
        line.extend(0,0,WIDTH, HEIGHT)



    # Render the frame
    renderer.render_frame(frame,"Grid")
renderer.close_video()


