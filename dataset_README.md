# MMVR: Millimeter-wave Multi-View Radar Dataset and Benchmark for Indoor Perception

## Abstract

The **Millimeter-wave Multi-View Radar (MMVR)** dataset comprises **345K** multi-view radar frames collected from **25** human subjects over **6** different rooms. It includes **446K** annotated bounding boxes/segmentation instances, and **7.59 million** annotated keypoints to support three perception tasks: object detection, pose estimation, and instance segmentation in the image plane. 

The data structure and configuration settings, including radar-to-camera coordinate transformation parameters, are detailed below.


## Data Structure

MMVR consists of 35 data sessions, each stored in a **data folder** with the format **$\text{d}x\text{s}y$**, where $x$ represents the day and $y$ the session index of that day when the data were collected. 

Within each data folder, data frames are grouped into one-minute **data segments**. Each data segment is stored in a folder named using a three-digit, zero-filled convention based on the chronological order. For example, the first one-minute data segment is saved in the folder **$000$**, while the second one-minute data segment is stored in the folder **$001$**.

Within each data segment folder, each **data frame** consists of 5 NPZ files: **meta**, **radar**, **bounding boxes** (bbox), **keypoints** (pose) and **segmentation masks** (mask). Each data frame is named using a five-digit, zero-filled convention based on the chronological order within the one-minute data segment. 

```
Root/
├── d1s1/
│   └── ...
├── d1s2/
│   ├── 000/
│   │   ├── 00000_meta.npz          ... Meta info
│   │   ├── 00000_radar.npz         ... Horizontal/Vertical heatmaps
│   │   ├── 00000_bbox.npz          ... 2D Bounding Boxes
│   │   ├── 00000_pose.npz          ... 2D keypoints
│   │   ├── 00000_mask.npz          ... 2D Segmentation masks
│   │   ├── 00001_meta.npz         
│   │   ├── 00001_radar.npz     
│   │   ├── 00001_bbox.npz      
│   │   ├── 00001_pose.npz      
│   │   ├── 00001_mask.npz      
│   │   .
│   │   ├── 00899_meta.npz          
│   │   ├── 00899_radar.npz         
│   │   ├── 00899_bbox.npz          
│   │   ├── 00899_pose.npz          
│   │   └── 00899_mask.npz          
│   └── 001/
│       ├── 00000_meta.npz          
│       ├── 00000_radar.npz        
│       ├── 00000_bbox.npz         
│       ├── 00000_pose.npz         
│       ├── 00000_mask.npz       
.       └── ...
├── d9s5/
│   └── ... 
└── d9s6/
    └── ...
```

## Data Frame: `XXXXX_*.npz`: 

Frame ID with zero padded at the beginning to 5 digits

`XXXXX_meta.npz` contains one NumPy arrays:
- `global_frame_id`: Global Frame Index, with dtype: `int`. 

`XXXXX_radar.npz` contains two NumPy arrays:
- `hm_hori`: Horizontal view heatmap, with shape:`(256, 128)` and dtype:`float32`. The range of first axis corresponding to depth in the radar coordinate is [0, 8] meters and the range of second axis corresponding to azimuth in the radar coordinate is [-1, 4] meters. The resolution of each pixel is the range of coordinates divided by the number of pixels.
- `hm_vert`: Vertical view heatmap, with shape:`(256, 128)` and dtype:`float32`. The range of first axis corresponding to depth in the radar coordinate is [0, 8] meters and the range of second axis corresponding to elevation in the radar coordinate is [-2, 3] meters. The resolution of each pixel is the range of coordinates divided by the number of pixels.

`XXXXX_bbox.npz` contains three NumPy arrays:
- `bbox_i`: 2D bounding box in the image plane with shape: `(n,5)` for `n` object with dtype:`float32` and `[x_min, y_min, x_max, y_max, score]` for each object:
    - `x_min`: x-coordinate (heigth) of the top-left corner of the bounding box.
    - `y_min`: y-coordinate (width) of the top-left corner of the bounding box.
    - `x_max`: x-coordinate (height) of the bottom-right corner of the bounding box.
    - `y_max`: y-coordinate (width) of the bottom-right corner of the bounding box.
    - `score`: confidence score of the detected object

- `bbox_hori`: 2D bounding box in the horizontal plane with shape: `(n,4)` for `n` object with dtype:`float32` and `[x_min, y_min, x_max, y_max]` for each object:
    - `x_min`: x-coordinate (depth) of the top-left corner of the bounding box.
    - `y_min`: y-coordinate (azimuth) of the top-left corner of the bounding box.
    - `x_max`: x-coordinate (depth) of the bottom-right corner of the bounding box.
    - `y_max`: y-coordinate (azimuth) of the bottom-right corner of the bounding box.

- `bbox_vert`: 2D bounding box in the vertical plane with shape: `(n,4)` for `n` object with dtype:`float32` and `[x_min, y_min, x_max, y_max]` for each object:
    - `x_min`: x-coordinate (depth) of the top-left corner of the bounding box.
    - `y_min`: y-coordinate (elevation) of the top-left corner of the bounding box.
    - `x_max`: x-coordinate (depth) of the bottom-right corner of the bounding box.
    - `y_max`: y-coordinate (elevation) of the bottom-right corner of the bounding box.

`XXXXX_pose.npz` contains one NumPy arrays:
- `kp`: 2D keypoints in the image plane with shape: `(n,17,3)` for `n` objects, ` 17` keypoints for each object with dtype:`float32`	and each keypoint represented by `3` elements:	
    - `x-coord`: x-coordinate (height) of one keypoint.
    - `y-coord`: y-coordinate (width) of one keypoint.
    - `score`: confidence score of one keypoint.
    - The order of 17 keypoints: `[Nose, Left Eye, Right Eye, Left Ear, Right Ear, Left Shoulder, Right Shoulder, Left Elbow, Right Elbow, Left Wrist, Right Wrist, Left Hip, Right Hip, Left Knee, Right Knee, Left Ankle, Right Ankle]`.

`XXXXX_mask.npz` contains one NumPy arrays:
- `mask`: 2D binary segmentation masks with shape `(n, 480, 640)` with `n` is the number of objects with dtype:`bool`.

## Python snippet

To load the data from the .npz files, you can use the following Python code snippet:

### load meta, radar, and image-plane annotation labels

```python
import numpy as np

root = 'root_path/d9s3/007/'
index = root + '00255'

# Meta info
data = np.load(index + '_meta.npz')
global_id = data['global_frame_id']
# Horizontal/Vertical heatmap
data = np.load(index + '_radar.npz')
hori = data['hm_hori']  # (256, 128)
vert = data['hm_vert']  # (256, 128)
# Bounding Box
data = np.load(index + '_bbox.npz')
bbox_i= data['bbox_i']  # (n, 5)
bbox_hori = data['bbox_hori']  # (n, 4)
bbox_vert = data['bbox_vert']  # (n, 4)
# keypoints
data = np.load(index + '_pose.npz')
kp = data['kp']  # (n, 17, 3)
# Segmentation mask
data = np.load(index + '_mask.npz')
mask = data['mask']  # (n, 480, 640)
```

### visualize radar heatmaps

```python
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 12))
ax= fig.add_subplot(1,2,1)
ax.imshow(hori)
ax.set_title('Horizontal radar heatmap')
ax= fig.add_subplot(1,2,2)
ax.imshow(vert)
ax.set_title('Vertical radar heatmap')
plt.show()
```

<table style="margin-left:auto;margin-right:auto;">
  <tr>
    <td style="text-align:center;">
      <img src="figs/heatmaps.png" alt="radar heatmaps" width="600"/><br>
    </td>
  </tr>
</table>

### visualize 2D image-plane annotation

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# joints connections for 2d keypoints
connections = np.array([[13, 15], [11, 13], [14, 16], [12, 14], [11, 12],
                [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
                [3, 5], [4, 6]])
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(1,1,1)
img = np.zeros((480, 640, 3), dtype=np.uint8)

for j in range(mask.shape[0]):
    # seg
    img[mask[j, :,:]] = (0, 255, 0)
    # bbox
    print(bbox_i[j,:4])
    x1, y1, x2, y2 = bbox_i[j,:4]
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title('bbox/keypoints/mask')
    # kp
    for connection in connections:
        x = kp[j,connection, 0]
        y = kp[j,connection, 1]
        ax.plot(x, y,color='b',marker='.')
ax.imshow(img)
plt.show()
```

<table style="margin-left:auto;margin-right:auto;">
  <tr>
    <td style="text-align:center;">
      <img src="figs/annotation.png" alt="annotation visualization" width="600"/><br>
    </td>
  </tr>
</table>

## MMVR Configuration Parameters

### The calibrated radar-to-camera coordinate transformation parameters are shown below

```python
# Rotation
R = np.array([[0.997214252753733, -0.0607987476962386, -0.0432116463859367],
                [-0.0630518006224704, -0.996610215186671, -0.0528445779985113],
                [0.0398522840384131, -0.0554219384733665, 0.997667381541953]])
# Translation
t = np.array([[-0.0648588235850897], [0.302966783204636], [0.0914051241089986]])
```

### The camera intrinsic parameters to project from 3D camera coordinate to 2D image plane. 

```python
fx, fy = 379.476, 379.476  # Focal lengths
ppx, ppy = 322.457, 241.627  # Principal point coordinates
k1, k2, k3 = -0.0566423, 0.0700418, -0.000190029  # Radial distortion coefficients
p1, p2 = 6.1314e-05, -0.0222783  # Tangential distortion coefficients
W, H = 640, 480  # image size
```

## MMVR Data Protocols

MMVR considers two protocols: **(P1)** a single subject in an open forground and **(P2)** multiple subjects in cluttered space. 

- P1 (single subject in an open foreground) includes $107.9$K data frames in a single open-foreground room with a single subject. These data were collected over 4 separate days (d1-d4) with one or two sessions per day. The subject walking and jumping in the space remains unobstructed to both radar and RGB camera observations. 

- P2 (multiple subjects in cluttered space) includes $237.8$K data frames in $5$ cluttered rooms with single and multiple subjects. Starting from Day 5 (d5), $6$ data sessions were collected in one room. During the data collection, the subjects were doing diverse activities such as walking, sitting, stretching, reading, writing on the board, and having conversations.

Within each data protocol, two data splits over data segments are considered: **(S1)** random split and **(S2)** cross-environment split. 

- S1 randomly splits the non-overlapping one-minute data segments ($122$ for P1 and $273$ for P2) into train, validation and test sets at a ratio of $80:10:10$.

- S2 first splits all data segments in d5, d6, d7, and d9 into train, validation, and test sets. Then, we include all data in d8 in the test set such that one can assess the generalization performance of trained model for an unseen environment (d8).

<table style="margin-left:auto;margin-right:auto;">
  <tr>
    <td style="text-align:center;">
      <img src="figs/dataProtocols.png" alt="MMVR data protocols" width="600"/><br>
    </td>
  </tr>
</table>

The data splits can be loaded using the `data_split.npz` file.

```python
split_file = 'data_split.npz'
data = np.load(split_file, allow_pickle=True)
data_split_dict = data['data_split_dict'].item()
P1S1 = data_split_dict['P1S1']
P1S2 = data_split_dict['P1S2']
P2S1 = data_split_dict['P2S1']
P2S2 = data_split_dict['P2S2']
```


## Citation

If you use this dataset, please cite our paper:
```
@inproceedings{MMVR2024,
  title={MMVR: Millimeter-wave Multi-View Radar Dataset and Benchmark for Indoor Perception},
  author={M. Mahbubur Rahman and Ryoma Yataka and Sorachi Kato and Pu Perry Wang and Peizhao Li and Adriano Cardace and Petros Boufounos},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  pages={},
  year={2024}
}
```

---
## License
```
Created by Mitsubishi Electric Research Laboratories (MERL), 2024.
SPDX-License-Identifier: CC-BY-SA-4.0
```