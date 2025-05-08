import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from IPython.display import display
import matplotlib.cm as cm


def create_diffusion_animation(frames_data, time_steps, true_values, total_steps=50, save_path="diffusion_steps", fps=2):
    """
    Create and save an animation from collected diffusion frames with true values overlaid.
    Multiple diffusion processes (from n_samples) are visualized with different colors.

    Parameters:
    - frames_data: List of arrays with diffusion steps - shape is [n_steps*n_samples, sequence_length]
    - time_steps: List of time step values
    - true_values: Array of true values to overlay
    - save_path: Path to save the GIF
    - fps: Frames per second for the animation
    """

    if not frames_data:
        print("No frames to animate.")
        return

    # Create figure
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes()

    # Determine the number of samples and frames per sample
    # Assuming all samples have the same number of time steps
    samples_with_unique_timesteps = {}
    for i, t in enumerate(time_steps):
        sample_idx = i // len(set(time_steps))  # Calculate which sample this frame belongs to

        if sample_idx not in samples_with_unique_timesteps:
            samples_with_unique_timesteps[sample_idx] = []

        # Store the frame index and time step
        samples_with_unique_timesteps[sample_idx].append((i, t))

    n_samples = len(samples_with_unique_timesteps)
    frames_per_sample = len(samples_with_unique_timesteps[0])
    mult_factor = total_steps // frames_per_sample

    print(f"Detected {n_samples} samples with {frames_per_sample} frames each")

    # Find global min and max for consistent y-axis limits
    all_diffusion_data = np.concatenate(frames_data)
    all_data = np.concatenate([all_diffusion_data, true_values])
    y_min, y_max = all_data.min(), all_data.max()
    # Add some padding
    y_range = y_max - y_min
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range

    # Generate colors for each sample using a colormap
    colors = cm.rainbow(np.linspace(0, 1, n_samples))

    # Keep track of final frames for each sample
    final_frames = {}

    def init():
        ax.set_xlim(0, len(frames_data[0]) - 1)
        ax.set_ylim(-1, 1)
        ax.set_title('Multiple Diffusion Processes with True Values')
        return []

    def update(frame_idx):
        ax.clear()

        # Plot the true values (constant across all frames)
        ax.plot(true_values, color='black', linewidth=3.0, label='True Values')

        # Calculate which sample and which frame within that sample
        current_sample = frame_idx // frames_per_sample
        frame_in_sample = frame_idx % frames_per_sample

        # For each sample that has completed or is in progress
        for sample_idx in range(min(current_sample + 1, n_samples)):
            # Get the appropriate frame for this sample
            if sample_idx == current_sample:
                # This is the current sample being processed - show its current frame
                sample_frame_idx = samples_with_unique_timesteps[sample_idx][frame_in_sample][0]
                current_time_step = samples_with_unique_timesteps[sample_idx][frame_in_sample][1]

                # Plot with lower opacity while in progress
                ax.plot(frames_data[sample_frame_idx], color=colors[sample_idx],
                        alpha=0.6, linewidth=1.5,
                        label=f'Sample {sample_idx + 1} (t={current_time_step * mult_factor})')

                # If this is the final frame for this sample, store it
                if frame_in_sample == frames_per_sample - 1:
                    final_frames[sample_idx] = frames_data[sample_frame_idx]
            else:
                # This sample has completed its diffusion - show its final frame
                if sample_idx in final_frames:
                    # Plot final frame with higher opacity
                    ax.plot(final_frames[sample_idx], color=colors[sample_idx],
                            alpha=0.8, linewidth=1.5,
                            label=f'Sample {sample_idx + 1} (final)')

        # Set title, limits, and legend
        overall_progress = f"Sample {current_sample + 1}/{n_samples}, Step {frame_in_sample + 1}/{frames_per_sample}"
        ax.set_title(f'Diffusion Process - {overall_progress}')
        ax.set_ylim(-1.1, 1)
        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Value')

        # Only show legend entries that are currently visible
        # Limit legend size if many samples
        # if n_samples > 5:
        #     ax.legend(loc='upper right', fontsize='small')
        # else:
        #     ax.legend(loc='best')

        return []

    # Create the animation
    total_frames = n_samples * frames_per_sample
    frames = list(range(total_frames))
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    # Save the animation
    print(f"Creating animation with {total_frames} total frames")
    ani.save(f'{save_path}.gif', writer='pillow', fps=fps)
    plt.savefig(f'{save_path}.png')
    plt.close(fig)

    print(f"Animation saved to {save_path}")

    # Display the animation if in a notebook environment
    try:
        from IPython.display import Image
        display(Image(filename=save_path))
    except:
        pass