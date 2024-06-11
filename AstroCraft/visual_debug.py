"""
This is a script to allow for visual debugging of the environment. What happens is the following:
    - An episode is played and rendered at each step. The state is also recorded at each time step.
    - The first frame is shown along with the state space. 
    - The user can step frame by frame through the episode with the left and right arrow keys.
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from ctfgymv3.CaptureTheFlag_v3 import CTFENV
import numpy as np

def display_image_and_text(images, texts):
    num_images = len(images)

    # Create the main Tkinter window
    root = tk.Tk()
    root.title("Image and Text Viewer")

    # Create a figure and axes for displaying the image
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Create a text widget for displaying the text
    text_widget = tk.Text(root, height=20, width=80)
    text_widget.pack(side=tk.LEFT, padx=10, pady=10)  # Adjust the padx and pady values as desired

    # Initialize the current index
    current_index = 0

    # Display the first image and text
    ax.imshow(images[current_index])
    ax.set_title(f"Frame #{current_index} of {num_images-1}")
    text_widget.insert(tk.END, texts[current_index])

    def update_display(event):
        nonlocal current_index

        if event.keysym == "Right" and current_index < num_images - 1:
            # Move to the next image and text
            current_index += 1
        elif event.keysym == "Left" and current_index > 0:
            # Move to the previous image and text
            current_index -= 1
        elif event.keysym == "Escape":
            # Close the window on pressing the Escape key
            root.quit()
            return

        # Clear the current image and text
        ax.cla()
        text_widget.delete(1.0, tk.END)

        # Display the new image and text
        ax.imshow(images[current_index])
        ax.set_title(f"Frame #{current_index} of {num_images-1}")
        text_widget.insert(tk.END, texts[current_index])

        # Update the canvas
        canvas.draw()

    def close_window():
        # Handle the window close event
        root.quit()

    # Bind the keypress event to the update_display function
    root.bind("<Key>", update_display)

    # Bind the window close event to the close_window function
    root.protocol("WM_DELETE_WINDOW", close_window)

    # Start the Tkinter event loop
    tk.mainloop()


def state_to_text(state, n_agents):
    """Converts the state space to a more readable format"""

    # Make blue text lines
    blue = f"Blue Base:\tOrbital: {np.round(state[0,-2],decimals=3)}\tAngle: {((10000*state[0,-1])//100)/100}\n"
    red = f"Red Base:\tOrbital: {np.round(state[n_agents+1,-2],decimals=3)}\tAngle: {((10000*state[n_agents+1,-1])//100)/100}\n"
    for i in range(1,n_agents+1):
        blue += f"Blue {i}:\tOrbital: {np.round(state[i,-2],decimals=3)}\tAngle: {((10000*state[i,-1])//100)/100}\tFuel: {np.round(state[i,3])}\tFlag: {bool(state[i,4])}\tTransferring: {bool(state[i,5])}\tIntercepting: {bool(state[i,6])}\n"
        red += f"Red {i}:\tOrbital: {np.round(state[n_agents+i+1,-2],decimals=3)}\tAngle: {((10000*state[n_agents+i+1,-1])//100)/100}\tFuel: {np.round(state[n_agents+i+1,3])}\tFlag: {bool(state[n_agents+i+1,4])}\tTransferring: {bool(state[n_agents+i+1,5])}\tIntercepting: {bool(state[n_agents+i+1,6])}\n"

    return blue + "\n" + red

    


if __name__ == "__main__":
    n_agents = 1
    env = CTFENV(n_agents, 'RandomPlayer')
    state, info = env.reset()
    frames = [env.render()]
    states = [state_to_text(state, n_agents)]
    print("Starting...")
    while True:
        # for agent in env._player0 + env._player1:
        #     print(f"Agent {agent._num} on team {agent._team} is at {agent._posvector}")
        action = env.action_space.sample(info['mask']['p0_mask'])
        print(action, info['mask']['p0_mask'])
        state, rew, term, trunc, info = env.step(action)
        frames.append(env.render())
        print(f"Simulated {len(frames)} turns")
        states.append(state_to_text(state, n_agents))
        print(env._player0[1]._posvector)
        
        if term or trunc:
            print("\n", end="")
            print("Done!")
            break

    # Display the images and texts
    display_image_and_text(frames, states)
