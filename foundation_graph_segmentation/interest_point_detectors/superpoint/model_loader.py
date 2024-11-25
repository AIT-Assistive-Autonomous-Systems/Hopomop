import torch
import numpy as np
from interest_point_detectors.superpoint.superpoint import SuperPoint


def superpoint_model_loader(max_length = -1, force_cpu = False, resize = [512,512], superglue = 'outdoor', nms_radius = 4, keypoint_threshold = 0.005, max_keypoints = 1024, sinkhorn_iterations = 20, match_threshold = 0.2):
    if len(resize) == 2 and resize[1] == -1:
        resize = resize[0:1]
    if len(resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            resize[0], resize[1]))
    elif len(resize) == 1 and resize[0] > 0:
        print('Will resize max dimension to {}'.format(resize[0]))
    elif len(resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if max_length > -1:
        pairs = pairs[0:np.min([len(pairs), max_length])]


    # Load the SuperPoint and SuperGlue models.
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
    }

    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
    return superpoint