import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import utils
import cv2
import numpy as np

def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map, score_map):

    desc = 0

    keypoint_map = keypoint_map.t().detach().numpy()
    keypoints = [cv2.KeyPoint(p[0], p[1], 1) for p in keypoint_map]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def read_image(img_file, image_size=(240,320)):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, image_size)
    img_orig = img.copy()

    img = utils.cv2_to_torch(img)
    return img, img_orig


def main(config):
    logger = config.get_logger('test')

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    img1, img1_orig = read_image('data/test/boat/img1.pgm')
    img1 = img1.unsqueeze(0).to(device)
    img2, img2_orig = read_image('data/test/boat/img4.pgm')
    img2 = img2.unsqueeze(0).to(device)

    A = model.unsuperpoint(img1)
    B = model.unsuperpoint(img2)

    AS, AP, AF = A['S'], A['P'], A['F']
    BS ,BP, BF = B['S'], B['P'], B['F']

    AS = AS.squeeze(0).to('cpu')
    AP = AP.squeeze(0).to('cpu')
    AF = AF.squeeze(0).to('cpu')

    BS = BS.squeeze(0).to('cpu')
    BP = BP.squeeze(0).to('cpu')
    BF = BF.squeeze(0).to('cpu')
    
    kp1, desc1 = extract_superpoint_keypoints_and_descriptors(AP, AF, AS)
    kp2, desc2 = extract_superpoint_keypoints_and_descriptors(BP, BF, BS)
  

    result1 = cv2.drawKeypoints(img1_orig, kp1, outImage = None, color=(255,0,0))
    cv2.imwrite("test5.jpg", result1)

    result2 = cv2.drawKeypoints(img2_orig, kp2, outImage = None, color=(255,0,0))
    cv2.imwrite("test6.jpg", result2)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
