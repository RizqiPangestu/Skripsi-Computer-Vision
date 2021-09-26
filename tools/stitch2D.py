import cv2
import numpy as np
import os
from collections import OrderedDict
import argparse
import time

"""
PARAMETER
"""
image_path = 'dataset/images'
output_path = 'dataset/stitched'
avg_time = 0
# scale for input image
work_megapix = 0.6
# scale for seaming image
seam_megapix = 0.1
# compose image
compose_megapix = -1
# confidence for blending between image
conf_thresh = 0.3
# bundle adjustment
ba_refine_mask = 'xxxxx'
wave_correct = None

# transform blend methods
warp_type = 'affine'
blend_type = 'multiband'
blend_strength = 0.05

# output
save_graph = True
result_name = None

# matcher finder
finder = cv2.ORB.create()
seam_work_aspect = 1
matcher_type = 'affine'
features = 'orb'

# use gpu
try_cuda = False
# Use OpenCL
cv2.ocl.setUseOpenCL(False)

# confidence for matching points similarity
match_conf = 0.3

# Compensator
expos_comp_type = cv2.detail.ExposureCompensator_GAIN_BLOCKS
expos_comp_nr_feeds = 1
expos_comp_block_size = 32


parser = argparse.ArgumentParser(
    prog="stitch2D.py", description="Stitching frame"
)
parser.add_argument(
    '--frames', action='store', default=5, type=int, dest='file_count'
)
parser.add_argument(
    '--res', action='store', default=1024, type=int, dest='resolution'
)

def get_matcher():
    matcher = cv2.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    return matcher

def get_compensator():
    compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp_type)
    return compensator


def main(files):
    img_names = files
    global result_name
    if result_name is None:
        result_name = "result" + img_names[0]
    
    print("==================================")
    print("[START]")
    time_start = time.perf_counter()
    print("[INPUT]",img_names,result_name)
    
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    
    for name in img_names:
        if name is result_name:
            full_img = cv2.imread(os.path.join(output_path,name) + ".jpg")
        else :
            full_img = cv2.imread(os.path.join(image_path,name) + ".jpg")
            full_img = cv2.resize(src=full_img, dsize=(resolution,resolution), interpolation=cv2.INTER_AREA)
        
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        # cv2.imshow("ori",full_img)
        # cv2.imshow("resized",full_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        print(full_img.shape)
        
        if not is_work_scale_set:
            if work_megapix < 0:
                img = full_img
                work_scale = 1
            else:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_work_scale_set = True
        if not is_seam_scale_set:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True

        img = cv2.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        img_feat = cv2.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        images.append(img)


    cv2.destroyAllWindows()

    matcher = get_matcher()
    p = matcher.apply2(features)
    matcher.collectGarbage()

    if save_graph:
        with open('graph.txt', 'w') as fh:
            fh.write(cv2.detail.matchesGraphAsString(img_names, p, conf_thresh))

    # Remove image that confidence < conf_thresh
    indices = cv2.detail.leaveBiggestComponent(features, p, conf_thresh)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i, 0]])
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
    images = img_subset
    img_names = img_names_subset
    full_img_sizes = full_img_sizes_subset
    
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        return result_name

    estimator = cv2.detail_AffineBasedEstimator()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    # Bundle Adjustment
    adjuster = cv2.detail_BundleAdjusterAffinePartial()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()

    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    if wave_correct is not None:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv2.detail.waveCorrect(rmats, wave_correct)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]

    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator = get_compensator()
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = cv2.detail_DpSeamFinder('COLOR')
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None
    
    # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
    for idx, name in enumerate(img_names):
        if name is result_name:
            full_img = cv2.imread(os.path.join(output_path,name) + ".jpg")
        else:
            full_img = cv2.imread(os.path.join(image_path,name) + ".jpg")
            full_img = cv2.resize(src=full_img, dsize=(resolution,resolution), interpolation=cv2.INTER_AREA)
        

        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

        if abs(compose_scale - 1) > 1e-1:
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            img = full_img

        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
    
        if blender is None:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv2.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv2.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])

    result_img = None
    result_mask = None
    result_img, result_mask = blender.blend(result_img, result_mask)
    cv2.imwrite(os.path.join(output_path,result_name) + '.jpg', result_img)
    time_end = time.perf_counter()
    global avg_time
    avg_time += time_end-time_start
    print(f"[DONE] in {time_end-time_start:0.4f} sec")
    return result_name,result_img.shape
    


if __name__ == '__main__':
    files = list()
    out_name = None
    args = parser.parse_args()
    file_count = args.file_count
    resolution = args.resolution
    print("frame/stitch =",file_count,"fps")

    for file in os.listdir(image_path):
        files.append(os.path.join(image_path,file))

    
    files.sort(key=lambda x: os.path.getmtime(x))

    for file in files:
        files[files.index(file)] = os.path.splitext(os.path.split(file)[1])[0]

    print("Total Images =",len(files),"images")
    print("match conf =",match_conf, "conf_thresh =",conf_thresh, "blend_strengh =",blend_strength)

    total_time_start = time.perf_counter()
    for i in range(len(files)-1):
        try:
            if out_name is None:
                out_name,_ = main([files[i],files[i+1]])
            else:
                out_name,out_shape = main([out_name,files[i+1]])
        except Exception as e:
            print("[EXCEPTION]",e)
    total_time_end = time.perf_counter()

    print("==================================")
    print("[OUTPUT] DIR =",os.path.join(output_path,out_name) + '.jpg')
    print("[OUTPUT] RES =",result_name, out_shape)
    print(f"[OUTPUT] Average stitching time = {avg_time/(file_count-1):0.4f}")
    print(f"[FINISHED] in {total_time_end-total_time_start:0.4f} sec")
    cv2.destroyAllWindows()