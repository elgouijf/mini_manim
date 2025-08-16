import os

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
        if not vmobject.submobjects:
            if not vmobject.subpaths:
                if vmobject.points.shape[0] == 0:
                    return#no points to draw

                points = vmobject.points
                for i in range(len(points)):
                    if not i:
                        self.ctx.move_to(*points[0])
                    else:
                        self.ctx.set_source_rgba(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                        vmobject.stroke_color[2],vmobject.opacity)
                        self.ctx.line_to(*points[i])
                        self.ctx.set_line_width(2)
                        self.ctx.stroke()
                if vmobject.close:
                    self.ctx.line_to(*points[0])
                
                    #fill options
                    self.ctx.set_source_rgb(vmobject.fill_color[0],vmobject.fill_color[1],
                                    vmobject.fill_color[2])#fill_color
                    self.ctx.fill_preserve()#fill but keep the path
                    self.ctx.set_source_rgba(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                    vmobject.stroke_color[2],vmobject.opacity)
                    self.ctx.paint()
                    self.ctx.set_line_width(2)
                    self.ctx.stroke()
            else:
                for i,subpath in enumerate(vmobject.subpaths):
                    if subpath.shape[0] == 0:
                        continue#no points to draw

                    points = subpath
                    for j in range(len(points)):
                        if not j:
                            self.ctx.move_to(*points[0])
                        else:
                            self.ctx.set_source_rgba(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                        vmobject.stroke_color[2],vmobject.opacity)
                            self.ctx.line_to(*points[j])
                            self.ctx.set_line_width(2)
                            self.ctx.stroke()
                    if vmobject.closed_subpaths[i]:
                        self.ctx.line_to(*points[0])
                    
                        #fill options
                        self.ctx.set_source_rgb(vmobject.fill_color[0],vmobject.fill_color[1],
                                        vmobject.fill_color[2])#fill_color
                        self.ctx.fill_preserve()#fill but keep the path
                        self.ctx.set_source_rgba(vmobject.stroke_color[0],vmobject.stroke_color[1],
                                        vmobject.stroke_color[2],vmobject.opacity)
                        self.ctx.paint()
                        self.ctx.set_line_width(2)
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

        self.render_vm(triangle)
    def render_text(self, text_obj) :
        ctx = self.ctx
        ctx.select_font_face(text_obj.font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(text_obj.font_size)
        ctx.set_source_rgb(text_obj.text_color[0], text_obj.text_color[1], text_obj.text_color[2])
        x, y = text_obj.position 
        ctx.move_to(x, y)
        ctx.show_text(text_obj.text)
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
    def render_text(self,text_obj:mbj.VMobject):
        """
        Render text object using Cairo
        """
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



class Scene:
    """"
    collection of objects to draw
    """
    def __init__(self,renderer):
        self.renderer = renderer
        self.mobjects = []
    def add(self,mobject):
        #storing the list pf visuals
        self.mobjects.append(mobject)
    def remove(self,mobject):

        for i in range(len(self.mobjects)):
            if mobject == self.mobjects[i]:
                self.mobjects.pop(i)
                break
    def render(self):
        for mobject in self.mobjects:
            self.renderer.render_polygon(mobject)


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
