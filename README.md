# SAM-CropSampleGeneration
## Background
This is an exploration on what [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) can do for remote sensing involving medium-resolution satellite imagery. Specifically, a sample generation approach based on SAM is proposed to solve sample scarcity problem in crop mapping.The workflow is shown in the picture. More details can be found in our ongoing paper. We hope our little contribution can offer some references on utlizing foundation models to solve remote sensing problems.


![](https://github.com/Nick0317Sun/SAM-CropSampleGeneration/blob/main/pics/workflow.png)


## Installation
We assume you already have understandings of Anaconda and PyTorch. Only the instructions of some specific required components are provided below. 
We utilized shapely, geopandas to manipulate shapefile files; rasterio to manipulate raster files; scipy and fastdtw to perform sample cleaning.

1. Enter your environment.

2. Install required packages:

`pip install -r requirements.txt`  

 or use conda to install, but fastdtw may not be properly installed.

`conda install --yes --file requirements.txt`

3. Download SAM [checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints), we used [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth). As you finish download, put the checkpoint in the same folder as main.py.

## Demo
We provided a simple demo of using Sentinel-2 time series and field-survey samples from [AAFC](https://open.canada.ca/data/en/dataset/503a3113-e435-49f4-850c-d70056788632) to generate more samples.

You can use commands below to run the code. Considering not everyone can have resources to execute SAM on GPU, you can also try our demo by using CPU. The processing time depends on your equipments and data used.

GPU:

`python main.py --optimalImage ./images/20190919.tif --patch_x 200 --patch_y 200 --device cuda`

CPU:

`python main.py --optimalImage ./images/20190919.tif --patch_x 200 --patch_y 200`

## Note
The raster and vector data we used have been processed according to our requirements, and our code may only be applicable to these processed data. 
If you intend to use our code with your own data, please review our code and make modifications specific to your data format, such as how to read the temporal information of rasters and attribute tables of vector files. If you encounter any difficulties, feel free to ask the main author of the code through jialinsun4815162342@gmail.com. We are delighted to engage in any academic and code-related discussions.

## License
[Facebook Research Segment Anything](https://github.com/facebookresearch/segment-anything) — Apache-2.0 license

## Citation
Still in progress.

## Some other awesome projects using SAM for remote sensing applications
[segment-anything-eo](https://github.com/aliaksandr960/segment-anything-eo)

[segment-geospatial](https://github.com/opengeos/segment-geospatial)

[samrs](https://github.com/vitae-transformer/samrs)
