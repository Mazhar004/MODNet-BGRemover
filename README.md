# MODNet_BGRemover

## Application

A deep learning approach to remove background and adding new background image

- Remove background from **images,videos & live webcam**
- Adding new background to those **images,videos & webcam footage**

## Installation

### Python Version

- Python == 3.8 (Any version of Python3 will work fine)

### Library Installation

#### Windows

- Library Install
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`

#### Linux

- Virtual Environment
  - `python -m venv venv`
  - `source venv/bin/activate`
- Library Install
  - `pip install --upgrade pip`
  - `pip install --upgrade setuptools`
  - `pip install -r requirements.txt`

## Inference

### Demo

<table style="text-align:center;font-size:bold">
<tr>
<td><b>Before removing the background</b></td>
<td><b>After removing the background</b></td>
</tr>
<tr>
<td><img src="assets/sample_image/female.jpeg" alt="Female.jpg" width="460" height="500"/></td>
<td><img src="output/female.jpeg" alt="Female.jpg" width="460" height="500"/></td>
</tr>
<tr>
<td><b>Before removing the background</b></td>
<td><b>After replacing the background with new image</b></td>
</tr>
<tr>
<td><img src="assets/sample_image/male.jpeg" alt="Male.jpg" width="460" height="500"/></td>
<td><img src="output/male.jpeg" alt="Male.jpg" width="460" height="500"/></td>
</tr>
<tr>
<td><b>Before removing the background from video</b></td>
<td><b>After replacing the background with new image in this video</b></td>
</tr>
<tr>><img src="assets/sample_image/sample.gif" alt="Video" width="460" height="500"/></td>
<td colspan=2><img src="output/sample.gif" alt="Video" width="920" height="400"/></td>
</tr>
<table>

### Image

#### Single image

It will generate the output file in **output/** folder

- `python inference.py --image image_path` **[Without background image]**
- `python inference.py --image image_path --background True` **[With background image]**
- Example:
  - `python inference.py --image assets/sample_image/female.jpeg`
  - `python inference.py --image assets/sample_image/male.jpeg --background True`

#### Folder of images

It will generate the output file in **output/** folder

- `python inference.py --folder folder_path` **[Without background image]**
- `python inference.py --folder folder_path --background True` **[With background image]**
- Example:
  - `python inference.py --folder assets/sample_image/`
  - `python inference.py --folder assets/sample_image/ --background True`

### Video

It will generate the output file in **output/** folder

- `python inference.py --video video_path` **[Without background image]**
- `python inference.py --video video_path --background True` **[With background image]**
- Example:
  - `python inference.py --video assets/sample_video/sample.mp4`
  - `python inference.py --video assets/sample_video/sample.mp4 --background True`

### Webcam

- `python inference.py --webcam True` **[Without background image]**
- `python inference.py --webcam True --background True` **[With background image]**
