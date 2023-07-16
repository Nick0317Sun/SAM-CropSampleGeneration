import argparse

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from utilis.utilis import *
from utilis.visualize import *
from utilis.SITS_utilis import *
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
from shapely import geometry
import pandas as pd
from time import time

parser = argparse.ArgumentParser(description='Sampling generation based on SAM')

parser.add_argument('--imagesDir', help='The directory where images located', default='./images')
parser.add_argument('--saveDir', help='The save directory where outputs located', default='./experiments')
parser.add_argument('--optimalImage', help='Manually allocate the input for SAM', default='None')
parser.add_argument('--patch_x', type=int, help='Manually set the size of the patch', default='None')
parser.add_argument('--patch_y', type=int, help='Manually set the size of the patch', default='None')
parser.add_argument('--shapefile', help='The shapefile contains your pixel-wise samples', default='None')
parser.add_argument('--device', help='Run SAM on gpu or cpu', default='cpu')

if __name__ == '__main__':
    t1 = time()

    args = parser.parse_args()

    # derive parameters
    images_dir = args.imagesDir
    save_dir = args.saveDir
    optimal_image = args.optimalImage
    patch_x, patch_y = args.patch_x, args.patch_y
    shp_path = args.shapefile
    device = args.device

    # initialize SAM
    sam_checkpoint = r"sam_vit_h_4b8939.pth"
    device = device
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # select an optimal image for SAM
    if optimal_image == 'None':  # did not allocate image, enter automatic image selection process
        optimal_image = ImageSelection(images_dir)

    # derive the coordinates and labels of samples
    if shp_path == 'None':
        shp_path = glob(images_dir + '/*.shp')[0]
    input_points, input_labels, input_ids = extract_coord_inTif(shp_path, optimal_image)

    # read optimal image for SAM, and convert to 0-255
    ds = rio.open(optimal_image)
    tif_image = ds.read([1, 2, 3])
    tif_image = tif_image.transpose(1, 2, 0)
    uint8_image = []
    for i in range(3):
        temp_data = tif_image[:, :, i]
        normalize = np.interp(temp_data, (np.min(temp_data), np.max(temp_data)), (0, 255)).astype(np.uint8)
        uint8_image.append(normalize)
    uint8_image = np.stack(uint8_image, axis=-1)

    # read shapefile contains samples
    gpd_samples = gpd.read_file(shp_path)

    final_shp = gpd.GeoDataFrame()  # set to save new final samples

    # begin the whole generating process
    for i, point in tqdm(enumerate(input_points), ncols=10):  # loop to generate new samples based on each reference sample
        # derive ROI, i.e., the patch, fed to SAM
        start_point, ROI, ROI_input_point = extract_ROI(point, uint8_image, size_x=patch_x, size_y=patch_y)
        vis_pointInMap(ROI, ROI_input_point, input_labels[i], os.path.join(save_dir, f'{input_ids[i]}_BeforeSAM.png'))

        # feed image to SAM
        predictor.set_image(ROI)
        masks, scores, logits = predictor.predict(
            point_coords=ROI_input_point[:, ::-1],
            point_labels=np.array([input_labels[i]]),
            multimask_output=True,
        )

        # visualize mask outputted by SAM
        for j, (mask, score) in enumerate(zip(masks, scores)):
            vis_MaskInMap(ROI, ROI_input_point, input_labels[i],
                          mask, j, score,
                          os.path.join(save_dir, f'{input_ids[i]}_AfterSAM_mask_{j + 1}.png'))

        # save Mask 1 as a sample candidate area
        coords_masks = np.argwhere(masks[0] == True)  # derive coordinates of pixels within the Mask 1
        coords_masks_origin = []
        for coord in coords_masks:
            x, y = coord
            x = x + start_point[0]
            y = y + start_point[1]
            coords_masks_origin.append([x, y])
        coords_masks_origin = np.array(coords_masks_origin)
        gpd_point = gpd_samples[gpd_samples['id'] == input_ids[i]]
        new_points = []
        for temp_point in coords_masks_origin:
            lon, lat = ds.xy(temp_point[0], temp_point[1])
            new_points.append({'random': 0, 'type': 'SAM', 'class': 0,
                               'multi_clas': input_labels[i], 'id': 0,
                               'geometry': geometry.Point(lon, lat)})
        if len(new_points) == 0:
            continue
        new_points_gdf = gpd.GeoDataFrame(new_points, crs=gpd_samples.crs)
        updated_points = gpd.GeoDataFrame(pd.concat([gpd_point, new_points_gdf], ignore_index=True), crs=gpd_samples.crs)
        temp_output = os.path.join(save_dir, 'temp_output.shp')
        updated_points.to_file(temp_output)

        # sample cleaning process by leveraging time series information
        temp_output = os.path.join(save_dir, 'temp_output.shp')
        shp, df = GetSortedDate(images_dir, SAM=True, shp_path=temp_output)
        type_mark, id_mark, label, sample = extractDatacube(shp, df, ['band1', 'band2', 'band3', 'band4'])
        time_node = df['dates'].to_list()
        interpolated_sample = timeSeriesConstruction(time_node, sample)
        result = sampleDf_toNumpy(interpolated_sample)
        id_mark = np.array(id_mark)
        type_mark = np.array(type_mark)

        # calculate NDVI and visualize
        red = result[:, :, 2]
        nir = result[:, :, 3]
        NDVI = (nir - red) / (nir + red)
        SAM_samples = NDVI[np.where(type_mark == 'SAM')]  # samples generated by SAM
        origin_sample = NDVI[np.where(type_mark != 'SAM')]  # reference sample
        vis_SITS(SAM_samples, os.path.join(save_dir, f'{input_ids[i]}_AfterSAM_SITS.png'), reference=origin_sample)

        SAM_samples[np.where(np.isnan(SAM_samples) == True)] = 0
        SAM_samples[np.where(np.isinf(SAM_samples) == True)] = 0
        origin_sample[np.where(np.isnan(origin_sample) == True)] = 0
        origin_sample[np.where(np.isinf(origin_sample) == True)] = 0

        # cleaning
        cleaned_samples, filtered_index = data_clean(SAM_samples, origin_sample, similarity_threshold=0.7)
        vis_SITS(cleaned_samples, os.path.join(save_dir, f'{input_ids[i]}_AfterSAM_SITS_cleaned.png'), reference=origin_sample)

        # delete samples that do not meet our requirements
        temp_shp = gpd.read_file(temp_output)
        filtered_index = np.array(filtered_index) + 1  # first line is reference sample, SAM samples begin from the second line
        temp_shp.drop(filtered_index, inplace=True)

        # delete medium temp files or not
        final_shp = pd.concat([final_shp, temp_shp])
        delete_files = glob(save_dir + '/temp_output*')
        # for file in delete_files:
        #     os.remove(file)
    final_shp.reset_index(drop=True, inplace=True)
    final_shp = final_shp.drop_duplicates(subset='geometry')  # delete repeated samples
    final_shp.to_file(os.path.join(save_dir, 'Generated_samples.shp'))

    t2 = time()
    time_elapsed = t2 - t1
    print(f'time consuming: **{time_elapsed // 60:.0f}m {time_elapsed % 60:.2f}s** or **{time_elapsed:.2f}s**')
    print('done')
