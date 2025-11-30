# Contact-GraspNet (PyTorch) -- 6-DoF Grasp Generation

This project provides an **inference-only setup** for the PyTorch
reimplementation of **Contact-GraspNet** (CGN), based on:

**Wen Tao Yuan et al., "Contact-GraspNet: Efficient 6-DoF Grasp
Generation in Cluttered Scenes", ICRA 2021.**\
Original TensorFlow code:
https://github.com/wentaoyuan/contact_graspnet\
PyTorch port used here: https://github.com/Xrenyao/cgn-pytorch\
PyPI package: `pip install cgn-pytorch`

The model takes a 3D point cloud and predicts:

-   **Grasp poses** (4×4 homogeneous matrices)\
-   **Confidence scores** (0--1)\
-   **Gripper widths** (meters)

Optional per-point RGB features are supported.

------------------------------------------------------------------------

## 🚀 Installation Guide (PyTorch + CUDA 12)

``` bash
mkdir ~/grasp6d_infer && cd ~/grasp6d_infer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### 1. Install PyTorch (CUDA 12.4 wheels)

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install CGN (PyTorch port) + dependencies

``` bash
# Install CGN without pulling conflicting deps
pip install cgn-pytorch==0.4.3 --no-deps

# Numpy / SciPy / Open3D / OpenCV / TQDM
pip install numpy==1.26.2 scipy==1.11.4 scikit-learn==1.3.2     open3d==0.19.0 opencv-python==4.10.0.84 tqdm

# Mesh + utilities
pip install importlib-resources==6.1.1 trimesh==4.10.0 psutil==5.9.6

# PyTorch Geometric stack (compatible wheels)
pip install torch-geometric==2.4.0             torch-cluster==1.6.3             torch-scatter==2.1.2             torch-sparse==0.6.18             torch-spline-conv==1.2.2
```

------------------------------------------------------------------------

## 🧪 Quick Sanity Test (CGN loads + CUDA availability)

``` bash
python - <<'PY'
import torch, cgn_pytorch
print("Torch:", torch.__version__)
m, _, _ = cgn_pytorch.from_pretrained()
print("CGN loaded successfully, CUDA:", torch.cuda.is_available())
PY
```

Expected:

    Torch: 2.x.x
    CGN loaded successfully, CUDA: True

------------------------------------------------------------------------

## 🧪 Minimal Synthetic Test (no RealSense required)

Run:

``` bash
python test_cgn_dummy.py
```

Expected output:

    synthetic pts: (20000, 3)
    Initializing net...
    ...net initialized.
    ✅ Got 50 grasps ≥0.6 confidence

------------------------------------------------------------------------

## 📥 Model Input

**Required:** - `pts`: Point cloud array shaped **\[N, 3\]**

**Optional:** - `rgb`: Per-point RGB values shaped **\[N, 3\]**

------------------------------------------------------------------------

## 📤 Model Output

Running:

``` python
poses, scores, widths = inference(model, pts, rgb)
```

Returns:

  Name       Shape       Description
  ---------- ----------- ---------------------------------------------
  `poses`    (K, 4, 4)   6-DoF grasp poses as homogeneous transforms
  `scores`   (K,)        Confidence score ∈ \[0, 1\]
  `widths`   (K,)        Gripper width in **meters**

------------------------------------------------------------------------

## 🔧 Implementation Details

The main inference logic is in:

    ~/.venv/lib/python3.12/site-packages/cgn_pytorch/inference.py

The PyTorch port is a near-faithful translation of the original
TensorFlow model from Stanford/Toyota CSRL.

------------------------------------------------------------------------

## ❗ Common Issues & Fixes

  -------------------------------------------------------------------------------------------------
  Issue                                  Fix
  -------------------------------------- ----------------------------------------------------------
  **TypeError: new(): data must be a     Use `inference(model, pts)`, not
  sequence (got dict)**                  `model.infer_pointcloud()`

  **torch-cluster / torch-scatter build  Install PyTorch **first**, then install the wheels as
  errors**                               listed above

  **Open3D window not appearing**        Do **not** run with `sudo`, ensure an active X11/Wayland
                                         session

  **RealSense Not Found**                Install:
                                         `sudo apt install librealsense2-utils librealsense2-dev`

  **Conflicting numpy / SciPy versions** Use pinned versions: `numpy==1.26.2`, `scipy==1.11.4`
  -------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 📚 Citation

If you use this in research:

    @inproceedings{yuan2021contact,
      title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
      author={Yuan, Wen Tao and Mousavian, Arsalan and Fox, Dieter},
      booktitle={ICRA},
      year={2021}
    }

------------------------------------------------------------------------

## 🔗 Useful Links

-   **Original TensorFlow repo:**\
    https://github.com/wentaoyuan/contact_graspnet\
-   **PyTorch port:**\
    https://github.com/Xrenyao/cgn-pytorch\
-   **Paper:**\
    https://arxiv.org/abs/2103.14127
