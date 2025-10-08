import os
from utilities.color import *
import sys
import cairo



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mobjects.mobjects as mbj
import numpy as np

import cv2
import subprocess

WIDTH = 480
HEIGHT = 360


class Renderer:
    #object to draw
    def __init__(self,im_surface,ctx,fps = 10,width =WIDTH,height = HEIGHT,file_name = None,ffmpeg_flag = False):
        self.im_surface = im_surface
        self.ctx = ctx
        self.fps = fps
        self.width = width
        self.height = height
        self.file_name= file_name
        if file_name:
            if not ffmpeg_flag:
                self.video_writer = cv2.VideoWriter(
                    file_name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (width,height)
                )
            else:
                self.ffmpeg_cmd = [
                    'ffmpeg',
                    '-y',  # overwrite output
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-pix_fmt', 'bgr24',  # or 'rgb24' depending on your array
                    '-s', f'{width}x{height}',
                    '-r', str(fps),
                    '-i', '-',  # input from stdin
                    '-an',  # no audio
                    '-vcodec', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    'output.mp4'
                ]
                            
    def render_polygone(self,mobject ):
        if mobject.points.shape[0] == 0:
            return#no points to draw
        #drawing succesion of points
        points = mobject.points
        for i in range(len(points)):
            if not i:
                self.ctx.move_to(*points[0])
            else:
                self.ctx.line_to(*points[i])
        self.ctx.line_to(*points[0])
        #fill options
        self.ctx.set_source_rgb(mobject.fill_color[0],mobject.fill_color[1],
                           mobject.fill_color[2])#fill_color
        self.ctx.fill_preserve()#fill but keep the path
        self.ctx.set_source_rgb(mobject.stroke_color[0],mobject.stroke_color[1],
                           mobject.stroke_color[2])
        self.ctx.set_line_width(2)
        self.ctx.stroke()
    #if it is  a vmobject
    def render_vm(self,vmobject : mbj.VMobject):
        
        if isinstance(vmobject, mbj.Text):
            

            ctx = self.ctx
       
       
            # If we have no pre-rendered surface, generate it now.
            # Choose target_pixel_width: for example text_obj.font_size * scale_factor
            if not hasattr(vmobject, "_surface") or vmobject._surface is None:
                # choose a width roughly proportional to font_size
                approx_w = int(vmobject.font_size * 40)  # tweak this factor as needed
                vmobject.generate_points_latex(target_pixel_width=approx_w)

            surf = vmobject._surface
            if surf is None:
                return

            # Positioning:
            # We treat text_obj.position as center by default. If you want top-left, change accordingly.
            x_center, y_center = vmobject.position

            # If your cairo coordinate origin is top-left (usual), and you want the surface centered:
            x = x_center - surf.get_width() / 2
            y = y_center - surf.get_height() / 2
            print(f"Rendering text at ({x}, {y}) with size ({surf.get_width()}x{surf.get_height()})")

            ctx.save()
            # if you need to flip y or scale, do it here. For usual top-left coords, no flip.
            # draw surface
            ctx.set_source_surface(surf, x, y)
            ctx.paint()   # paint with full alpha
            ctx.restore()
            return
        
        if not vmobject.submobjects:
            if not vmobject.subpaths:
                if vmobject.points.shape[0] == 0:
                    return#no points to draw
                self.ctx.new_path()
                points = vmobject.points
                for i in range(len(points)):
                    if not i:
                        self.ctx.move_to(*points[0])
                    else:
                      
                        self.ctx.line_to(*points[i])
                      
                if vmobject.closed:
                    self.ctx.line_to(*points[0])
                
                #fill options
                r,g,b = vmobject.fill_color
                self.ctx.set_source_rgba(r,g,b,vmobject.fill_opacity)
                self.ctx.fill_preserve()
                #stroke options 
                r,g,b = vmobject.stroke_color
                self.ctx.set_source_rgba(r,g,b,vmobject.stroke_opacity)
                self.ctx.stroke()
            else:
                for i, subpath in enumerate(vmobject.subpaths):
                    if subpath.shape[0] == 0:
                        continue#no points to draw
                    self.ctx.new_path()
                    points = subpath
                    for j in range(len(points)):
                        if not j:
                            self.ctx.move_to(*points[0])
                        else:
                             self.ctx.line_to(*points[j])
                    if vmobject.closed_subpaths[i]:
                        self.ctx.line_to(*points[0])
                r,g,b = vmobject.fill_color.get_rgb()
                
                self.ctx.set_source_rgba(r,g,b,vmobject.opacity)
                self.ctx.fill_preserve()

                r,g,b = vmobject.stroke_color.get_rgb()
                self.ctx.set_source_rgba(r,g,b,vmobject.opacity)
                self.ctx.set_line_width(1)
                self.ctx.stroke()
                   
                    
                       
        else:
            for submobject in vmobject.submobjects:
                if isinstance(submobject,mbj.VMobject):
                    self.render_vm(submobject)
                else:
                    continue
      
    def render_arrow2d(self,arrow : mbj.Arrow2d):
        end_point = arrow.offset + arrow.tip
        tip = arrow.tip
        self.ctx.move_to(*arrow.offset)
        self.ctx.line_to(*end_point)
        triangle = mbj.VMobject()
        triangle.points = np.array([
            [end_point[0]+0.1*tip[0],end_point[1] + 0.1*tip[1]],
            [end_point[0] + (0.01)*(1/2 *tip[0]  + np.sqrt(3)/2 *tip[1]),
             end_point[1] + 0.01*(1/2 * tip[1] -np.sqrt(3)/2 *tip[0])],
            [end_point[0] + 0.01*(1/2 *tip[0]  - np.sqrt(3)/2 *tip[1]),
             end_point[1] + 0.01*(1/2 * tip[1] +np.sqrt(3)/2 *tip[0])]
        ])
        triangle.close()
        triangle.set_fill_color((1, 0, 0))
        triangle.stroke_color = (1,0,0) 

    def render_line(self,line : mbj.Line):
        """
        Render a line object using Cairo
        """
        self.ctx.move_to(*line.start_pt)
        self.ctx.line_to(*line.end_pt)
        self.ctx.set_source_rgb(line.stroke_color[0], line.stroke_color[1], line.stroke_color[2])
        self.ctx.set_line_width(1)
        self.ctx.stroke()
    def render_frame(self,frame_index,out_name,main_save = True):
        buffer = self.im_surface.get_data()#getting the pixels of the image
        frame = np.ndarray(shape=(self.height,self.width,4),dtype=np.uint8,buffer=buffer)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) #convert cairo to opencv
        if main_save:
            cv2.imwrite(f"{out_name}/frame_{frame_index}.png",frame)
        self.video_writer.write(frame)  
    def close_video(self):
        if self.video_writer:
            self.video_writer.release()
            print("video saved succesfully")
    def render_text(self,text_obj:mbj.Text,cairo_format = cairo.FORMAT_ARGB32,WIDTH=WIDTH,HEIGHT=HEIGHT):
        """
        Render text object using Cairo
        """
        if not text_obj.use_latex:
            if text_obj.opacity is not None and text_obj.opacity < 1.0:
           
                ctx = self.ctx
                ctx.select_font_face(text_obj.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(text_obj.font_size)
                ctx.set_source_rgba(text_obj.text_color[0], text_obj.text_color[1], text_obj.text_color[2], text_obj.opacity)
                x, y = text_obj.position 
                extents = ctx.font_extents()
                y = y + extents[0]  # Adjust y position for baseline
                ctx.move_to(x, y)
                ctx.show_text(text_obj.text)
            else:
                ctx = self.ctx
                ctx.select_font_face(text_obj.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(text_obj.font_size)
                ctx.set_source_rgb(text_obj.text_color[0], text_obj.text_color[1], text_obj.text_color[2])
                x, y = text_obj.position 
                extents = ctx.font_extents()
                y = y + extents[0] # Adjust y position for baseline
                ctx.move_to(x, y)
                ctx.show_text(text_obj.text)

        else:
            ctx = self.ctx
            ctx.select_font_face(text_obj.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(text_obj.font_size)
            if text_obj.opacity is not None and text_obj.opacity < 1.0:
                ctx.set_source_rgba(text_obj.text_color[0], text_obj.text_color[1], text_obj.text_color[2], text_obj.opacity)
            else:
                ctx.set_source_rgb(text_obj.text_color[0], text_obj.text_color[1], text_obj.text_color[2])
            
           
            # Generate LaTeX points once
            text_obj.generate_points_latex()

            text_obj.closed_subpaths = [True] * len(text_obj.subpaths)  # Ensure all subpaths are closed
            
          
            temp_surface = cairo.ImageSurface(cairo_format, WIDTH, HEIGHT)
            
           
            temp_ctx = cairo.Context(temp_surface)
            temp_ctx.set_source_rgba(1, 1, 1, 0)  # transparent background
            temp_ctx.paint()


            renderer.ctx = temp_ctx
            renderer.render_vm(text_obj)  # draw text once
            renderer.ctx = ctx  # restore main context
            text_obj.temp_surface = temp_surface
            print("Temporary surface created for LaTeX rendering.")

    def render(self, mobject):

        """
        Render a mobject using the appropriate method based on its type.
        """
        if isinstance(mobject, mbj.VMobject):
            self.render_vm(mobject)
        elif isinstance(mobject, mbj.Arrow2d):
            self.render_arrow2d(mobject)
        elif isinstance(mobject, mbj.Text):
            self.render_text(mobject)
        elif isinstance(mobject, mbj.Line):
            self.render_line(mobject)
        else:
            self.render_polygone(mobject)


class Scene:
    """"
    collection of objects to draw
    """
    def __init__(self,renderer, fps = 60):
        self.renderer = renderer
        self.mobjects = []
        self.frame = 0
        self.fps = fps

    def add(self, *mobject):
        #storing the list pf visuals
        self.mobjects.extend(mobject)

    def remove(self, *mobjects):
        for mobj in self.mobjects:
            if mobj in mobjects:
                self.mobjects.remove(mobj)
                

    def render_frame(self, out="output", main_save=True):
        self.renderer.ctx.set_source_rgb(0, 0, 0) 
        self.renderer.ctx.paint()
        for mobject in self.mobjects:
            self.renderer.render(mobject)
        self.renderer.render_frame(self.frame, out, main_save)
        self.frame += 1

    
    def wait(self, duration, out="output", main_save=True):
        """
        Freeze the current frame for `duration` seconds.
        """
        frames_to_wait = int(duration * self.fps)
        for _ in range(frames_to_wait):
            self.render_frame(out, main_save=main_save)

    def play(self, animation, out_name="output", main_save=True):
        """
        Play an animation on the scene.
        """
        duration = animation.duration
        frames = int(duration * self.fps)

        for frame in range(frames):
            t = frame / (frames - 1) if frames > 1 else 1
            animation.interpolate(t)
            animation.mobject.apply_transform()  # Apply the transformation to the mobject
            """ print("Frame", frame, "first point:", animation.mobject.points[0]) """
            self.render_frame(out_name, main_save=main_save)
        animation.finish()  # Ensure the final state is applied
        animation.mobject.apply_transform()  # Apply the final transformation to the mobject

def test_mobject_basic_transform():
    mobj = mbj.Mobject()
    mobj.points = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    
    # Set visual style
    mobj.fill_color = (0.2, 0.6, 0.8)     # light blue
    mobj.stroke_color = (0, 0, 0)         # black
    
    # Apply transformations
    mobj.move_to(200, 150)
    mobj.apply_transform()
    mobj.scale(80)
    mobj.apply_transform()

    # Cairo setup
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    renderer = Renderer(surface, ctx)
    scene = Scene(renderer)

    scene.add(mobj)
    scene.render()

    surface.write_to_png("mobject_output.png")
    print("Saved to mobject_output.png")
#test_mobject_basic_transform()
def test_vmobject_open_and_closed():
 

    # Cairo setup
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    # adjust import if needed
    renderer = Renderer(surface, ctx)

    # Create a closed shape
    closed_vm = mbj.VMobject()
    closed_vm.points = np.array([
        [100, 100],
        [150, 100],
        [150, 150],
        [100, 150]
    ])
    closed_vm.fill_color = (0.3, 0.6, 0.3)
    closed_vm.stroke_color = (0, 0, 0)
    closed_vm.close()

    # Create an open shape
    open_vm = mbj.VMobject()
    open_vm.points = np.array([
        [200, 200],
        [250, 200],
        [250, 250]
    ])
    open_vm.fill_color = (0.8, 0.2, 0.2)
    open_vm.stroke_color = (0, 0, 0)
    open_vm.open()

    # Render both
    renderer.render_vm(closed_vm)
    renderer.render_vm(open_vm)

    surface.write_to_png("vmobject_test_output.png")
    print("Saved to vmobject_test_output.png")

#test_vmobject_open_and_closed()

def test_text() :
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    renderer = Renderer(surface, ctx)
    ctx.set_source_rgb(1,1,1)
    ctx.paint()
    text_obj = mbj.Text("Hello, Mini Manim !", (50,50), "Sans", 32, (0.2, 0.2, 0.7))
    renderer.render_text(text_obj)
    surface.write_to_png("test_text_output.png")
    print('Saved to test_text_output.png')

#test_text()
WIDTH, HEIGHT, FPS = 640, 360, 30

# Cairo setup
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
ctx = cairo.Context(surface)

# Create renderer with video recording
renderer = Renderer(surface, ctx, FPS,WIDTH, HEIGHT, file_name="output.mp4")
arr = mbj.Arrow2d(tip = np.array((100,0)),offset=np.array((WIDTH/2,HEIGHT/2)))
# Draw frames
for frame in range(60):
    ctx.set_source_rgb(0, 0, 0)  # background
    ctx.paint()

    #ctx.set_source_rgb(1, 0, 0)  # red circle
    #ctx.arc(320, 180, 50 + frame, 0, 2 * np.pi)
    renderer.render_arrow2d(arr)

    #renderer.render_arrow2d(mbj.Arrow2d.x_axis(0,0.9*WIDTH,HEIGHT/2))
   # renderer.render_arrow2d(mbj.Arrow2d.y_axis(0,0.9*HEIGHT,WIDTH/2))       
    arr.tip = np.array((arr.tip[0]*np.cos(np.pi/30) - arr.tip[1]*np.sin(np.pi/30),
                       arr.tip[1]*np.cos(np.pi/30) + arr.tip[0]*np.sin(np.pi/30) ))
    #ctx.fill()

    #renderer.render_frame()

#renderer.close_video()
