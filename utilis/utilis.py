import geopandas as gpd
from pyproj import CRS, Transformer
import rasterio as rio
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from glob import glob


def ImageSelection(tif_dir, quality_band=4):
    """
        tif_dir: images directory
        quality_band: points out the channel that contains cloud, shadow...
    """
    tifs = glob(tif_dir + '/*.tif')

    invalid_pixels = []
    for tif in tifs:
        ds = rio.open(tif)
        data = ds.read()
        quality_channel = data[quality_band, :, :]

        # quality_channel == k   ,   k denotes the pixel value of invalid pixels
        pixels = len(list(np.where((quality_channel == 8) | (quality_channel == 9) |
                                   (quality_channel == 10) | (quality_channel == 3) | (quality_channel == 0))[0]))
        invalid_pixels.append(pixels)
    image_index = np.argsort(invalid_pixels)[0]
    return tifs[image_index]


def extract_coord_inTif(shapefile, tif):
    """
    Extract the row and column numbers of point features in the grid based on vector point data and grid data
    """
    gpd_samples = gpd.read_file(shapefile)
    raster = rio.open(tif)

    input_point = []
    input_label = []
    input_ids = []
    for idx, point in gpd_samples.iterrows():
        x, y = point['geometry'].x, point['geometry'].y
        label = int(point['multi_clas'])
        id_ = id_ = int(point['id'])

        row, col = raster.index(x, y)

        input_point.append([row, col])
        input_label.append(label)
        input_ids.append(id_)

    input_point = np.array(input_point)
    input_label = np.array(input_label)
    input_ids = np.array(input_ids)

    return input_point, input_label, input_ids


def to_LonLat(shapefile):
    """
        Convert the projection coordinates of the shapefile to latitude and longitude coordinates
    """
    gdf = gpd.read_file(shapefile)

    input_crs = gdf.crs.to_epsg()
    output_crs = CRS.from_epsg(4326)

    transformer = Transformer.from_crs(input_crs, output_crs, always_xy=True)

    coords = []
    for idx, point in gdf.iterrows():
        x, y = point['geometry'].x, point['geometry'].y
        lon, lat = transformer.transform(x, y)
        coords.append([lon, lat])

    return coords


def extract_ROI(point, image, size_x=100, size_y=100, noExtraction=False):
    """
        Extract ROI or patch and update the position of coordinates in the newly extracted region
        point: coordinates,[x,y]
        image: image, [H,W,C]

    """
    if not noExtraction:
        x = point[0]
        y = point[1]
        # Calculate the starting and ending coordinates of the extracted region
        start_x = max(x - size_x // 2, 0)
        start_y = max(y - size_y // 2, 0)
        end_x = min(x + size_x // 2, image.shape[1])
        end_y = min(y + size_y // 2, image.shape[0])

        region = image[start_x:end_x, start_y:end_y, :]

        new_x = x - start_x  # sample coordinates in the new extracted region
        new_y = y - start_y

        return [start_x, start_y], region, np.array([[new_x, new_y]])
    else:
        return [0, 0], image, np.array([point])


def data_clean(data, reference, similarity_threshold=0.45):
    """
    根据FastDTW进行数据清洗
    :param data:  待清洗数据
    :param reference:  参考数据
    :param similarity_threshold:  相似性阈值
    :return:
    """
    x_axis = np.arange(1, data.shape[1] + 1)
    reference_data = np.column_stack((x_axis, reference[0, :]))

    filter_samples = []
    filteted_index = []  # 记录被过滤掉的样本的索引
    for i, sample in enumerate(data):
        sample_data = np.column_stack((x_axis, sample))
        distance, _ = fastdtw(sample_data, reference_data, dist=euclidean)  # 距离越小，相似度越高
        similarity = 1 / (1 + distance)
        if similarity > similarity_threshold:
            filter_samples.append(sample)
        else:
            filteted_index.append(i)
    filter_samples = np.array(filter_samples)
    return filter_samples, filteted_index
