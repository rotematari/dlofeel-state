import time
import h5py
import numpy as np
import viser
from pathlib import Path

from scipy.spatial.transform import Rotation as R
from dlofeel_state.data.dataset import _decode_image_1d

# --- Configuration ---
IMG_FILE_PATH = Path("results_eval/good/img/26_01_14_58_np32_sinencTrue_attTrue_normTrue_norminpTrue_poseFalse_diffTrue_blankrefTrue_img320x240_bs32_lr0.0001_wd0.0001_dp0.2_ep10/with_spline/04_02_15_56_np32_sinencTrue_attTrue_normTrue_norminpTrue_poseFalse_diffTrue_blankrefTrue_img320x240_bs32_lr0.0001_wd0.0001_dp0.2_ep10/preds_gts.h5")
POSE_FILE_PATH = Path("results_eval/good/pose/26_01_16_35_np32_sinencTrue_attTrue_normTrue_norminpTrue_poseTrue_diffTrue_blankrefTrue_img320x240_bs32_lr0.0001_wd0.0001_dp0.2_ep10/preds_gts.h5")
def ros_to_viser_quat(q):
    """Converts ROS [x, y, z, w] to Viser [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]])

def get_preds_vs_gt(file_path: Path):
    try:
        with h5py.File(file_path, 'r') as f:
            poses = f['poses'][:]        # [x, y, z, qx, qy, qz, qw]
            preds = f['preds'][:]        # (N, 12, 3)
            gts = f['gts'][:]            # (N, 12, 3)
            paths_raw = f['sample_paths'][:]
            losses = f['losses'][:]
        
            # Decode paths
            paths = [Path(p.decode('utf-8')) if isinstance(p, bytes) else Path(p) for p in paths_raw]
            
        return poses, preds, gts, paths, losses
    except Exception as e:
        print(f"Error reading {file_path.name}: {e}")
        return [], [], [], [], []


def main():
    # 1. Setup Viser Server
    server = viser.ViserServer()
    print("Viser server started. Check the URL printed above.")

    # 2. Load Data
    poses, preds, gts, paths, losses = get_preds_vs_gt(IMG_FILE_PATH)
    if len(poses) == 0:
        print("No data loaded. Exiting.")
        return

    # 3. GUI Elements
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Frame", min=0, max=len(poses) - 1, step=1, initial_value=0
        )
        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        stop_button = server.gui.add_button("Stop", icon=viser.Icon.PLAYER_STOP)
    
    gui_info = server.gui.add_text(label="", initial_value="loading...", multiline=True)

    # --- Digit Cameras GUI ---
    digit_image_widgets = []
    with server.gui.add_folder("Digit Cameras"):
        for i in range(4):
            # Placeholder image
            dummy_img = np.zeros((320, 240, 3), dtype=np.uint8)
            w = server.gui.add_image(dummy_img, format='png', label=f"Cam {i}")
            digit_image_widgets.append(w)

    # --- Scene Setup (Grid/Lights) ---
    server.scene.add_grid(
        "/ground", width=2.0, height=2.0, cell_size=0.1, plane="xy", position=(0.0, 0.0, 0.0),
    )
    server.scene.add_light_directional(
        "/direction_light", wxyz=(1.0, 0.0, 0.0, 0.0), position=(2, 2, 5), intensity=0.8, cast_shadow=True
    )

    # State for playback
    playing = False
    num_frames = len(poses)

    # 4. Update Function
    def update_scene(index):
        pose = poses[index]
        pred = preds[index]
        gt = gts[index]
        path = paths[index]
        loss = losses[index]

        # Update Text Info
        gui_info.value = f"Sample: {path.parent.name}/{path.name} \nLoss: {loss:.4f}"

        # --- 1. Visualize Images from Original Source File ---
        try:
            # We open the original sample file to get the images on the fly
            with h5py.File(path, 'r') as f_sample:
                if 'digit_images' in f_sample:
                    # digit_imgs_numpy = f_sample['digit_images'] # [4, 3, H, W]
                    imgs = []
                    for k in range(4):
                        ds = f_sample['digit_images'][f'img_{k}']
                        raw = ds[:]  # 1D uint8
                        h = int(ds.attrs.get('height', 320))
                        w = int(ds.attrs.get('width', 240))
                        step = int(ds.attrs.get('step', 70))
                        enc = str(ds.attrs.get('encoding', 'bgr8'))

                        decoded = _decode_image_1d(raw, h, w, step, enc)
                        # chw = _to_chw_float(decoded)
                        imgs.append(decoded)
                    # Transpose to [4, H, W, 3] for Viser
                    # digit_imgs_numpy = np.transpose(digit_imgs_numpy, (0, 2, 3, 1))
                    digit_imgs_numpy = np.stack(imgs, axis=0)
                    # Normalize/Cast
                    if digit_imgs_numpy.max() <= 1.0:
                        digit_imgs_numpy = (digit_imgs_numpy * 255.0).astype(np.uint8)
                    else:
                        digit_imgs_numpy = digit_imgs_numpy.astype(np.uint8)

                    # Update widgets
                    for i, widget in enumerate(digit_image_widgets):
                        widget.image = np.ascontiguousarray(digit_imgs_numpy[i])
        except Exception as e:
            print(f"Error loading images from {path}: {e}")

        # --- 2. Visualize EE Frame ---
        server.scene.add_frame(
            "/ee_pose",
            position=pose[:3],
            wxyz=ros_to_viser_quat(pose[3:7]),
            axes_length=0.15,
            axes_radius=0.005
        )

        # --- 3. Visualize Markers (Pred vs GT) ---
        N = pred.shape[0]

        # Predictions = Magenta (Matching "Resampled/Computed" style from dataset)
        server.scene.add_point_cloud(
            "/preds_markers",
            points=pred,
            point_size=0.008,
            point_shape="circle",
            colors=np.tile([255, 0, 255], (N, 1)) 
        )

        # Ground Truth = Yellow (Matching "Markers/Raw" style from dataset)
        server.scene.add_point_cloud(
            "/gts_markers",
            points=gt,
            point_size=0.008,
            point_shape="circle",
            colors=np.tile([0, 255, 255], (N, 1))
        )

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

    # 6. Main Loop
    while True:
        if playing and gui_timestep.value < num_frames - 1:
            gui_timestep.value += 1
            time.sleep(0.008) # ~125 FPS max (limited by GUI update speed usually)
        else:
            if playing and gui_timestep.value >= num_frames - 1:
                playing = False
            time.sleep(0.05)

if __name__ == "__main__":
    main()