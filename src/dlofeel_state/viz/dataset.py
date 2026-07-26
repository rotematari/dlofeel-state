import time
import numpy as np
import viser
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# import data set 
from dlofeel_state.data.dataset import DLOH5SampleDataset
# --- Configuration ---
# Point this to the specific session folder (e.g., .../ds_h5/153_50_16_11)
DATA_SET_DIR = Path("src/datasets/c1/ds/test")
# TEST_DATA_SET_DIR = Path("src/datasets/packed/test/test_packed.h5")
BLACK_REF_DIR_PATH = Path("src/datasets/ds/digit_blank_closed_refs")
CONFIG_PATH = Path("src/dlofeel_state/configs/default.yaml")
NORM_STATS_PATH = Path("src/dlofeel_state/configs/norm_stats.yaml")
NORMALIZE = False
NORMALIZE_INPUTS = False
USE_DIFF_IMAGES = True
USE_BLANK_REF = True
NUM_RESAMPLE_POINTS = 32
IS_PACKED = False
MAX_TIME_DELTA = 0.13
IMG_SIZE_H = 320
IMG_SIZE_W = 240
POINT_SIZE = 0.01
def ros_to_viser_quat(q):
    """Converts ROS [x, y, z, w] to Viser [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]])

def main():
    # 1. Setup Viser Server
    server = viser.ViserServer()
    print("Viser server started. Check the URL printed above.")

    # 2. Find and Sort Files
    # import dataset

    ds_resampled = DLOH5SampleDataset(root_dir=DATA_SET_DIR, 
                                    norm_stats_path=NORM_STATS_PATH,
                                    blank_ref_root=BLACK_REF_DIR_PATH,  
                                    normalize=NORMALIZE, 
                                    normalize_inputs=NORMALIZE_INPUTS,
                                    use_diff_images=USE_DIFF_IMAGES,
                                    use_blank_ref=USE_BLANK_REF,
                                    img_size_H=IMG_SIZE_H,
                                    img_size_W=IMG_SIZE_W, 
                                    transform=None,
                                    num_resample_points=NUM_RESAMPLE_POINTS,
                                    is_packed=IS_PACKED,
                                    past_window_steps=1,
                                    max_time_delta=MAX_TIME_DELTA,
                                    )
    ds = DLOH5SampleDataset(root_dir=DATA_SET_DIR, 
                                    norm_stats_path=NORM_STATS_PATH,
                                    blank_ref_root=BLACK_REF_DIR_PATH,  
                                    normalize=NORMALIZE, 
                                    normalize_inputs=NORMALIZE_INPUTS,
                                    use_diff_images=USE_DIFF_IMAGES,
                                    use_blank_ref=USE_BLANK_REF,
                                    img_size_H=IMG_SIZE_H,
                                    img_size_W=IMG_SIZE_W, 
                                    transform=None,
                                    num_resample_points=NUM_RESAMPLE_POINTS,
                                    is_packed=IS_PACKED,
                                    past_window_steps=1,
                                    max_time_delta=MAX_TIME_DELTA,
                                    )
    # 3. GUI Elements
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Frame", min=0, max=len(ds) - 1, step=1, initial_value=0
        )
        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        stop_button = server.gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)
    gui_info = server.gui.add_text(label="" ,initial_value="loading...",multiline=True)
    # with server.gui.add_folder("Digit Cameras"):
    # We create 4 placeholder images. We use a dummy black image initially.
    # Viser expects (H, W, C).
    digit_image_widgets = []
    with server.gui.add_folder("Digit Cameras"):
        for i in range(4):
            # Create a dummy 100x100 black image to reserve the spot
            dummy_img = np.zeros((320, 240, 3), dtype=np.uint8)
            w = server.gui.add_image(dummy_img, format='png', label=f"Cam {i}")
            digit_image_widgets.append(w)
            
    # State for playback
    playing = False
    num_frames = len(ds)
    # 4. Update Function
    def update_scene(index):
        
        sample = ds[index]
        sample_resampled = ds_resampled[index]

        digit_imgs_numpy = sample["digit_images"].numpy()  # [4,3,H,W]
        digit_imgs_numpy = np.transpose(digit_imgs_numpy, (0, 2, 3, 1))  # -> [4,H,W,3]
        # Normalize if data is 0.0-1.0
        if digit_imgs_numpy.max() <= 1.0+1e-5:
                digit_imgs_numpy = (digit_imgs_numpy * 255.0).astype(np.uint8)
        else:
                digit_imgs_numpy = digit_imgs_numpy.astype(np.uint8)
        for i, widget in enumerate(digit_image_widgets):
                # Update the image property directly
                widget.image = np.ascontiguousarray(digit_imgs_numpy[i])
        try:
            
            # --- Load Data ---
            ee_pos: np.ndarray = sample['ee_pose'].numpy() # (7,)
            
            markers: np.ndarray = sample['marker_positions'].numpy() # (12, 3)

            # --- Visualize EE Frame ---
            server.scene.add_frame(
                "/ee_pose",
                position=ee_pos[:3],
                wxyz=ros_to_viser_quat(ee_pos[3:]),
                axes_length=0.15,
                axes_radius=0.005
            )
            # --- Visualize Markers ---
            # Get the number of markers (N)
            N = markers.shape[0]

            # Create a color array of shape (N, 3)
            marker_colors = np.tile([0, 255, 255], (N, 1)) 
            

            server.scene.add_point_cloud(
                "/markers",
                points=markers,
                point_size=POINT_SIZE,
                point_shape="circle",
                colors=marker_colors  # Now matches the shape of 'points'
            )            
            server.scene.add_point_cloud(
                "/markers_resampled",
                points=sample_resampled['marker_positions'].numpy(),
                point_size=POINT_SIZE,
                point_shape="circle",
                colors=(255, 0, 255)  
            )
            # digit_imgs = sample["digit_images"].numpy()  # [4,3,H,W]
            # digit_imgs = np.clip(digit_imgs * 255.0, 0, 255).astype(np.uint8)
            # digit_imgs = np.transpose(digit_imgs, (0, 2, 3, 1))  # -> [4,H,W,3]
            # for widget, img in zip(digit_image_widgets, digit_imgs):
            #     widget.value = img
            
            path = Path(sample['path'])
            gui_info.value = f"Dir: {path.parent.name} \nSample :{path.name}"
        except Exception as e:
            print(f"Error reading {index}: {e}")

    # 5. Event Bindings
    @gui_timestep.on_update
    def _(_):
        update_scene(gui_timestep.value)

    @play_button.on_click
    def _(_):
        nonlocal playing
        playing = True
        
    @stop_button.on_click
    def _(_):
        nonlocal playing
        playing = False

    # Initial load
    update_scene(0)

    # 6. Main Loop (Handles Playback Here)
    while True:
        if playing and gui_timestep.value < num_frames - 1:
            gui_timestep.value += 1
            time.sleep(0.0001) # ~30 FPS
        else:
            # If not playing (or at end), sleep to reduce CPU usage
            # and check for the 'playing' flag again shortly.
            if playing and gui_timestep.value >= num_frames - 1:
                playing = False # Auto-stop at end
            time.sleep(0.00001)

if __name__ == "__main__":
    main()
