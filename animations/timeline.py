class TimeLine:
    """
    A class to represent a timeline of animations.
    
    Attributes:
        animations (list): A list of animations in the timeline.
    """

    def __init__(self):
        self.animations = []
        self.current_time = 0

    def add_animation(self, animation):
        """
        Adds an animation to the timeline.
        
        Args:
            animation: The animation to be added.
        """
        if not hasattr(animation, 'start_time') or not hasattr(animation, 'end_time'):
            raise ValueError("Animation must have 'start_time' and 'end_time' attributes.")
        self.animations.append(animation)

    def get_animations(self):
        """
        Returns the list of animations in the timeline.
        
        Returns:
            list: The list of animations.
        """
        return self.animations
    
    def update(self, dt):
        """
        Updates the current time of the timeline and applies animations that are active at this time.
        
        Args:
            dt (float): The time delta to update the timeline.
        """
        self.current_time += dt

        for animation in self.animations:
            if animation.start_time <= self.current_time <= animation.end_time:
                t = (self.current_time - animation.start_time)/ animation.duration
                rate_function = animation.rate
                f_t = rate_function(t)  
                animation.interpolate(f_t)

            elif self.current_time > animation.end_time:
                animation.finish()

            elif self.current_time < animation.start_time:
                continue

        self.animations = [anim for anim in self.animations if not anim.finished]

    def is_finished(self):
        """
        Checks if all animations in the timeline are finished.
        
        Returns:
            bool: True if all animations are finished, False otherwise.
        """
        if len(self.animations) == 0 :
            self.current_time = 0
            return True
        return False

