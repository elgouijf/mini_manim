import mobjects.mobjects as mbj
import manim_bg.renderer as rdr
import cairo
from utilities.color import BLACK, RED
import numpy as np
import utilities.color as cl
WIDTH, HEIGHT, FPS = 1920, 1080, 30

# Cairo setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

# Create renderer
renderer = rdr.Renderer(surface, ctx, FPS, WIDTH, HEIGHT, file_name="grid.mp4")

# Create grid
grid = mbj.Line.grid(WIDTH, HEIGHT, 15, 27)
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
    fr"""Grid Animation of $\begin{{bmatrix}} {a} & {b} \\ {c} & {d} \end{{bmatrix}}$
    of determinant """,
    position=(WIDTH/2,HEIGHT/10),
    color=RED.get_rgb(),
    font_size=50,
    ctx=ctx,
    use_latex=True
)
Text.surface = surface
Text.stroke_color = BLACK
Text.stroke_width = 2


Text.closed_subpaths = [True] * len(Text.subpaths)  # Ensure all subpaths are closed

renderer.render_text(Text)
det = mbj.Text(
    fr"""{a*d - b*c}""",
    position=(WIDTH/2 ,HEIGHT/10 + Text._surface.get_height()),  
    color=BLACK.get_rgb(),
    font_size=40,
    ctx=ctx,
    use_latex=False
)
det.surface = surface   
det.stroke_color = BLACK
det.stroke_width = 2
det.closed_subpaths = [True] * len(det.subpaths)  # Ensure all sub
e1 = np.array([132,0])
e2 = np.array([0,40])
center = np.array([WIDTH/2,HEIGHT/2])
quadrelateral = mbj.VMobject(
    points = np.array([
                      center,
                      center + e1,
                      center +e1 + e2,
                      center  + e2
    ]
                      ))
quadrelateral.close
quadrelateral.stroke_color = cl.RED
quadrelateral.stroke_width = 2
quadrelateral.fill_color = cl.RED

# draw raw subpaths (no scale/translate) as thin colored outlines so we can inspect raw coordinates



for frame in range(1000):
    # Clear main surface
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.paint()
    print(f"Rendering frame {frame+1}/1000")
    # Transform and draw grid
    matrix = np.array([
        [a*frame/1000 + (1-frame/1000), b*frame/1000],
        [c*frame/1000, d*frame/1000 + (1-frame/1000)]
    ])
    renderer.render_text(det)
    a_det = a*frame/1000 + (1-frame/1000)
    d_det = d*frame/1000 + (1-frame/1000)
    determinant = a_det * d_det - b * c * (frame/1000)**2
   
    det.text = fr"""{determinant:.2f}"""
    renderer.render_text(det)

    for line in grid:
        transformed_start = np.dot(reflct_mat, np.dot(matrix, np.dot(reflct_mat, starts[line]-np.array([WIDTH/2, HEIGHT/2]))))
        transformed_end   = np.dot(reflct_mat, np.dot(matrix, np.dot(reflct_mat, ends[line]-np.array([WIDTH/2, HEIGHT/2]))))
        line.start_pt = transformed_start + np.array([WIDTH/2, HEIGHT/2])
        line.end_pt   = transformed_end + np.array([WIDTH/2, HEIGHT/2])
        line.extend(0,0,WIDTH,HEIGHT)  # Ensure line is within bounds
        renderer.render_line(line)
    quadrelateral.points = np.array([
        center,
        center  + reflct_mat@matrix@reflct_mat@e1,
        center  + reflct_mat@matrix@reflct_mat@(e1+e2),
        center + reflct_mat@matrix@reflct_mat@e2
    ])
    renderer.render_vm(quadrelateral)
    # Blit pre-rendered text
    ctx.set_source_surface(Text.temp_surface, 0, 0)
    ctx.paint()

    # Save frame
    renderer.render_frame(frame, "Grid")

renderer.close_video()

