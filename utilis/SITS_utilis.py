import os
from glob import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from itertools import chain


def GetSortedDate(path, SAM=False, shp_path=None):
    raster_names = glob(path + f'/*.tif')
    if SAM:
        shp_file = shp_path  # shapefile contains potential samples generated by SAM
    else:
        shp_file = glob(path + '/*.shp')[0]  # original collected samples

    # derive the date when each image was shot
    temp = pd.DataFrame()
    dates = []
    for i, filename in enumerate(raster_names):
        filename = os.path.basename(filename)
        dates.append(filename[:8])  # this should be revised based on your own data name
    temp['dates'] = dates
    temp['files'] = raster_names
    temp.sort_values(by='dates', inplace=True, ascending=True)  # 按日期升序

    return shp_file, temp


def getFeatureRowCol(img, shp_info):
    info_list = []
    for i, point in enumerate(shp_info[:, 1]):
        row, col = img.index(point.x, point.y)
        info_list.append([row, col, shp_info[i, 0]])
    return info_list


def stackDatacube(df_raster):
    tif_list = df_raster['files'].to_list()
    # time_list = df_raster['dates'].to_list()
    datacube = [rio.open(raster).read().transpose(1, 2, 0) for raster in tif_list]
    datacube = np.array(datacube)
    return datacube


def quality_check(data):
    delete_index = []
    cloud = data[:, 4]  # 云量波段是第五波段
    delete_index.append(list(np.where((cloud == 8.0) | (cloud == 9.0) | (cloud == 10.0) | (cloud == 3.0))[0]))
    for i in range(4):
        t = data[:, i]
        delete_index.append(list(np.where(t == 0)[0]))
    delete_index = list(chain.from_iterable(delete_index))
    delete_index = np.array(delete_index, dtype=np.int64)
    delete_index = np.unique(delete_index)
    return delete_index


def extractSampleSITS(sample, datacube, time_step, bands):
    row = sample[0]
    col = sample[1]
    cropcode = sample[2]
    data = datacube[:, row, col, :]

    d_index = quality_check(data)
    time_step = np.array(time_step)
    data = np.delete(data, d_index, 0)
    time_step = np.delete(time_step, d_index)

    data = np.delete(data, 4, 1)

    df_sample = pd.DataFrame()
    df_sample['dates'] = pd.to_datetime(time_step, format='%Y%m%d')
    for i in range(data.shape[1]):
        df_sample[bands[i]] = data[:, i]
    return cropcode, df_sample


def extractDatacube(shp, tifs, bands):
    df_sample = []
    label = []

    raster_temp = tifs.iat[0, 1]
    with rio.open(raster_temp) as img:
        template_EPSG = img.crs.to_epsg()

    gpd_shp = gpd.read_file(shp)
    gpd_shp = gpd_shp.to_crs(epsg=template_EPSG)
    shp_info = gpd_shp[['multi_clas', 'geometry']].values

    id_mark = gpd_shp['id'].values

    type_mark = gpd_shp['type'].values

    newshp_info = getFeatureRowCol(img, shp_info)

    datacube = stackDatacube(tifs)

    for sample in newshp_info:
        cropcode, sample = extractSampleSITS(sample, datacube, tifs['dates'].to_list(), bands)
        label.append(cropcode)
        df_sample.append(sample)

    return type_mark, id_mark, label, df_sample


def timeSeriesConstruction(time_node, sample_list):
    '''
        linear interpolation
    '''
    interpolated_list = []
    for sample in sample_list:
        sample = sample.set_index(['dates'], drop=True)
        sample = sample.reindex(time_node)

        sample = sample.interpolate(method='linear', axis=0)
        sample.fillna(method='ffill', axis=0, inplace=True)
        sample.fillna(method='bfill', axis=0, inplace=True)

        sample = sample.reset_index()
        sample = sample.rename(columns={'index': 'dates'})

        interpolated_list.append(sample)

    return interpolated_list

def sampleDf_toNumpy(sample_list):
    result = []
    for i, sample in enumerate(sample_list):
        sample = sample.drop('dates', axis='columns')
        sample = np.array(sample)
        result.append(sample)
    result = np.array(result)
    return result


def save_to_npy(np_sample, np_label, np_id, np_type, save_path):
    sample_path = os.path.join(save_path, 'samples.npy')
    label_path = os.path.join(save_path, 'labels.npy')
    np.save(sample_path, np_sample)

    np_label = np.array(np_label)
    np.save(label_path, np_label.squeeze())

    np.save(os.path.join(save_path, r'id_mark.npy'), np_id.squeeze())
    np.save(os.path.join(save_path, r'type_mark.npy'), np_type.squeeze())
