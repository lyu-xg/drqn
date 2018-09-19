import gym
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

# Imports specifically so we can render outputs in Jupyter.
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display, clear_output


def display_frames_as_gif(frames):
    clear_output()
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))
    
    
def display_frame(frame):
    display_frames_as_gif([frame])

    
def show_img(img):
    print(img.shape)
    plt.imshow(img[0])
    plt.show()
    
