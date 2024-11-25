import numpy as np
import fpsample
import torch
from transformers import SamModel, SamProcessor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from model.utils.mahalanobis import mahalanobis_distance
# remove double points
def duplicate_voting(points, predictions, labels):

    # find double points and vote for class
    for p in np.unique(points, axis=0):
        ix = np.where((points == p).all(axis=1))
        if len(ix[0]) > 1:
            # vote for class
            votes = predictions[ix]
            predictions[ix] = np.bincount(votes).argmax()

    points, unique_ix = np.unique(points, axis=0, return_index=True)
    predictions = predictions[unique_ix]
    labels = labels[unique_ix]

    return points, predictions, labels

def mahalanobis_filter(points, predictions):
    mahalanobis_distances_full = np.zeros(len(predictions))

    # extract mahalanobis distances for each point of class
    for c in np.unique(predictions):
        if c == 0:
            continue
        ix = np.where(predictions == c)
        class_points = points[ix]
        
        if class_points.shape[0] < 3:
            continue

        mahalanobis_distances = mahalanobis_distance(class_points)
        mahalanobis_distances_full[ix] = mahalanobis_distances

    mahalanobis_distances_full = (mahalanobis_distances_full - np.min(mahalanobis_distances_full)) / (np.max(mahalanobis_distances_full) - np.min(mahalanobis_distances_full)+1e-6)
    return mahalanobis_distances_full
from sklearn.preprocessing import StandardScaler
def isolation_forest_filter(points, predictions, labels, features):
    to_delete = np.zeros(len(predictions), dtype=bool)

    for c in np.unique(predictions):
        ix = np.where(predictions == c)
        class_points = points[ix]

        features_ = features[0][ix]

        # # reduce dimensionality
        # scaler = StandardScaler()
        # features_ = scaler.fit_transform(features_)
        


        # class_points = np.concatenate([class_points, features_], axis=1)
    
        # also label concat
        #class_points = np.concatenate([class_points, labels[ix].reshape(-1, 1)], axis=1)

        # lof = LocalOutlierFactor(n_neighbors=20)

        
        # if class_points.shape[0] < 3:
        #     continue
        # outlier_pred = lof.fit_predict(class_points)

        # # concatenate points to features
        clf = IsolationForest(n_estimators=20, warm_start=True)
        
        clf.fit(class_points)
        
        outlier_pred = clf.predict(class_points)
        to_delete[ix] = outlier_pred == -1

    points = points[~to_delete]
    predictions = predictions[~to_delete]
    labels = labels[~to_delete]

    return points, predictions, labels

class Segmentor:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.fps = config["fps"]

    def segment(self, image, points, predictions, labels, scores, features):

        # lookup the score for each points and prediction, remove if score is below threshold
        # scores [2024,3]
        #prediction [2024]
        # prediction defines which score is taken out of the three 
        
        best_scores = scores[np.arange(len(predictions)), predictions]


        #cut all scores in the lower 20% but not if there are less than 5 points of the class
        
        to_delete = np.zeros(len(predictions), dtype=bool)
        for c in np.unique(predictions):
            if c == 0:
                continue
            ix = np.where(predictions == c)
            class_points = points[ix]
            class_scores = best_scores[ix]
            if len(class_points) < 5:
                continue
            threshold = np.percentile(class_scores, 1)
            to_delete[ix] = class_scores < threshold

        points = points[~to_delete]
        predictions = predictions[~to_delete]
        labels = labels[~to_delete]

        # remove double points
        points, predictions, labels = duplicate_voting(points, predictions, labels)

        #points, predictions, labels = isolation_forest_filter(points, predictions, labels, features)

        mahalanobis_distances = mahalanobis_filter(points, predictions)


        # filter points with mahalanobis distance

        # filter points with mahalanobis distance

        point_ix = np.where(mahalanobis_distances > self.config["point_threshold"])[0]
        sam_point_positions = np.delete(points, point_ix, axis=0)
        sam_point_classes = np.delete(predictions, point_ix)
        sam_point_labels = np.delete(labels, point_ix)

        box_ix = np.where(mahalanobis_distances > self.config["box_threshold"])[0]
        sam_box_positions = np.delete(points, box_ix, axis=0)
        sam_box_classes = np.delete(predictions, box_ix)
        sam_box_labels = np.delete(labels, box_ix)


        boxes = {}
        for c in np.unique(sam_box_classes):
            ix = np.where(sam_box_classes == c)
            class_points = sam_box_positions[ix]
            min_x = np.min(class_points[:, 0])
            max_x = np.max(class_points[:, 0])
            min_y = np.min(class_points[:, 1])
            max_y = np.max(class_points[:, 1])
            boxes[c] = ([min_x, min_y, max_x, max_y])



        sam_point_samples = self.config["sam_point_samples"]
        sam_promp_type = self.config["sam_promt_type"]
        # generate sam masks
        iou_scores, masks, mask_order = self.get_sam_masks(image, sam_point_positions, boxes, sam_point_classes, promp_type=sam_promp_type, sam_point_samples=sam_point_samples, fps=self.fps)

        if not isinstance(iou_scores, type(None)):
            predicted_mask = self.merge_final_mask(masks, iou_scores, mask_order, image.shape[:2]).astype(np.uint8)
        else:
            predicted_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return predicted_mask, sam_point_positions, sam_point_classes
        

    def get_sam_masks(self, image, points, boxes, predictions, promp_type = "pointbox", sam_point_samples = 10, fps = False):
        all_points_list = []
        all_labels_list = []
        all_boxes_list = []
        mask_order = []
        unique_classes, count = np.unique(predictions, return_counts=True)

        
        for class_index in enumerate(unique_classes):
            if class_index[1] == 0:
                continue

            mask_index = np.where(predictions == class_index[1])            
            mask_order.append(class_index[1])
            
            positions = points[mask_index]

            all_boxes_list.append(boxes[class_index[1]])

            # CURRENTLY I JUST USE ALL POINTS, THIS NEEDS IMPROVEMENT
            if fps:
                #samples = min(sam_point_samples, len(positions))
                # fill up with random points
                if len(positions) < sam_point_samples:
                    # saple 10 random points with replace
                    positions = positions[np.random.choice(positions.shape[0], sam_point_samples, replace=True)]
                    
                
                position_ids = fpsample.fps_sampling(positions, sam_point_samples)
                positions = positions[position_ids]
            else:
                positions = positions[np.random.choice(positions.shape[0], sam_point_samples, replace=True)]
       
    
            all_points_list.append(positions)
       
            # first len(position) are positive, rest is negative
            labels = [1 for f in range(len(positions))]

            all_labels_list.append(labels)
        
        if len(all_points_list) == 0:
            return None, None, None
            
        if promp_type == "pointbox":
            inputs = self.sam_processor(image, input_points=[all_points_list], input_labels=[all_labels_list], input_boxes=[all_boxes_list], return_tensors="pt").to(self.device)
        elif promp_type == "point":
            inputs = self.sam_processor(image, input_points=[all_points_list], input_labels=[all_labels_list], return_tensors="pt").to(self.device)
        elif promp_type == "box":
            inputs = self.sam_processor(image, input_boxes=[all_boxes_list], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam_model(**inputs)
        
        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        
        return outputs.iou_scores[0], masks[0], mask_order


    def merge_final_mask(self, masks, iou_scores, classes_to_mask, shape):
        mask_list = []
        for mask_id, mask in enumerate(masks):
            sorted_indices = np.argsort(iou_scores[mask_id].detach().cpu().numpy()) 
            best_mask = mask[sorted_indices[2]]
            mask_list.append(best_mask)

        # sort mask by size # from big to small
        mask_sizes = [np.count_nonzero(mask) for mask in mask_list]
        order_mask = np.argsort(mask_sizes)[::-1]
        # get the right class for each mask
        final_mask = np.zeros(shape, dtype=np.uint8)
        for i, mask_id in enumerate(order_mask):
            final_mask[mask_list[mask_id]] = classes_to_mask[mask_id]

        return final_mask
        

        




        