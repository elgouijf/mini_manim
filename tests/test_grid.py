import mobjects.mobjects as mbj
import manim_bg.renderer as rdr
import cairo
from utilities.color import BLACK, RED
import numpy as np

WIDTH, HEIGHT, FPS = 1920, 1080, 30

# Cairo setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

# Create renderer
renderer = rdr.Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="grid.mp4")

# Create grid
grid = mbj.Line.grid(WIDTH, HEIGHT, 20, 20)
for line in grid:
    line.stroke_color = (0, 0, 0)
    line.stroke_width = 1

# Ask for transformation matrix values
a = float(input("Enter a: "))
b = float(input("Enter b: "))
c = float(input("Enter c: "))
d = float(input("Enter d: "))

starts = {line: line.start_pt for line in grid}
ends = {line: line.end_pt for line in grid}
reflct_mat = np.array([[1, 0], [0, -1]])

# Create LaTeX Text
Text = mbj.Text(
    fr"""Grid Animation of $\begin{{bmatrix}} {a} & {b} \\ {c} & {d} \end{{bmatrix}}$""",
    position=(WIDTH/2, HEIGHT/2),
    color=RED.get_rgb(),
    font_size=50,
    use_latex=True
)
Text.stroke_color = BLACK
Text.stroke_width = 2

# Generate LaTeX points once
Text.generate_points_latex()

Text.closed_subpaths = [True] * len(Text.subpaths)
print("Number of subpaths:", len(Text.subpaths))
for i, sp in enumerate(Text.subpaths[:5]):  # first 5 subpaths
    print(f"Subpath {i} length:", len(sp)) 
# Pre-render LaTeX to a temporary surface
temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
temp_ctx = cairo.Context(temp_surface)
temp_ctx.set_source_rgba(1, 1, 1, 0)  # transparent background
temp_ctx.paint()


renderer.ctx = temp_ctx
renderer.render_vm(Text)  # draw text once
renderer.ctx = ctx  # restore main context
print("Number of subpaths:", len(Text.subpaths))
for i, sp in enumerate(Text.subpaths[:6]):
    print(f"Subpath {i} len={len(sp)} bbox={sp.min(axis=0)}->{sp.max(axis=0)}")
# --- Animation loop ---
temp_surface_debug = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
temp_ctx_debug = cairo.Context(temp_surface_debug)
# draw raw subpaths (no scale/translate) as thin colored outlines so we can inspect raw coordinates

print("wrote raw_all.png")

for frame in range(1000):
    # Clear main surface
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.paint()

    # Transform and draw grid
    matrix = np.array([
        [a*frame/1000 + (1-frame/1000), b*frame/1000],
        [c*frame/1000, d*frame/1000 + (1-frame/1000)]
    ])
    for line in grid:
        transformed_start = np.dot(reflct_mat, np.dot(matrix, np.dot(reflct_mat, starts[line]-np.array([WIDTH/2, HEIGHT/2]))))
        transformed_end   = np.dot(reflct_mat, np.dot(matrix, np.dot(reflct_mat, ends[line]-np.array([WIDTH/2, HEIGHT/2]))))
        line.start_pt = transformed_start + np.array([WIDTH/2, HEIGHT/2])
        line.end_pt   = transformed_end + np.array([WIDTH/2, HEIGHT/2])
        line.extend(0,0,WIDTH,HEIGHT)  # Ensure line is within bounds
        renderer.render_line(line)

    # Blit pre-rendered text
    ctx.set_source_surface(temp_surface, 0, 0)
    ctx.paint()

    # Save frame
    renderer.render_frame(frame, "Grid")

renderer.close_video()

