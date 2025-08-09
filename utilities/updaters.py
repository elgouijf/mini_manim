import mobjects.mobjects as mbj
import numpy as np

up = "up"
down = "down"
right = "right"
left = "left"

def rotate_continuous(mobj,  dt, ang_speed = np.pi/4):
    """ ang_speed = #angle per second """
    mobj.rotate(ang_speed * dt)


def move_continuous(mobj,direction, dt, lin_speed = 3):
    center = mobj.get_center()
    x , y = center

    if direction == up :
        mobj.move_to(x, y + lin_speed * dt)

    elif direction == down :
        mobj.move_to(x, y - lin_speed * dt)

    elif direction == right :
        mobj.move_to(x + lin_speed * dt, y)

    elif direction == left :
        mobj.move_to(x - lin_speed * dt, y)


def scale_continous(mobj, dt, scale_speed = 0.3):
    mobj.scale(1 + scale_speed * dt)


def oscillate_continous(mobj, dt, direction, frequency = 2, amplitude = 1, oscill_speed = 2): #spped covers the area between two local extremums in 1 second
    if not hasattr(mobj, "_start_point"):
        mobj._start_point = mobj.get_center()

    if not hasattr(mobj, "oscill_time_r"): # this checks for attributes and give it to the instance in argument if it does'nt have them
        mobj._oscill_time_r = 0

    if not hasattr(mobj, "oscill_time_l"): 
        mobj._oscill_time_l = 0
    
    if not hasattr(mobj, "oscill_time_u"): 
        mobj._oscill_time_u = 0
    
    if not hasattr(mobj, "oscill_time_d"): 
        mobj._oscill_time_d = 0

    x0, y0 = mobj._start_point

    if direction == right :   
        mobj._oscill_time_r += dt # to the next position ( we have to do all of these calculations since trigonometrical functions aren't linear)
        off = amplitude * np.sin(2*np.pi * frequency * mobj._oscill_time_r)

        mobj.move_to(x0 + off, y0)

    elif direction == left :
        mobj._oscill_time_l += dt # to the next position ( we have to do all of these calculations since trigonometrical functions aren't linear)
        off = amplitude * np.sin(2*np.pi * frequency * mobj._oscill_time_l)

        mobj.move_to(x0 - off, y0)

    elif direction == up :
        mobj._oscill_time_u += dt # to the next position ( we have to do all of these calculations since trigonometrical functions aren't linear)
        off = amplitude * np.sin(2*np.pi * frequency * mobj._oscill_time_u)

        mobj.move_to(x0, y0 + off)

    elif direction == down :
        mobj._oscill_time_d += dt # to the next position ( we have to do all of these calculations since trigonometrical functions aren't linear)
        off = amplitude * np.sin(2*np.pi * frequency * mobj._oscill_time_d)

        mobj.move_to(x0, y0 - off)

