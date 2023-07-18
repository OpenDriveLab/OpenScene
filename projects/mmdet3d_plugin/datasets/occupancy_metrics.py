"""
Part of the code is taken from https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/sscMetrics.py
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = np.zeros(_C, dtype=np.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = np.sum(iou[1:]) / np.count_nonzero(cnt_class[1:])
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    target = np.int32(target)
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = np.argmax(
        predict, axis=1
    )  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = np.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc

class SSCMetrics:
    def __init__(self, n_classes, eval_far=True, eval_near=True,
                 near_distance=10, far_distance=30, occ_type='normal'):
        """
        non-empty class: 0-15
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
        'driveable_surface', 'other_flat', 'sidewalk',
        'terrain', 'manmade,', 'vegetation'
        """
        self.n_classes = n_classes
        self.empty_label = n_classes
        self.foreground_obj_num = 10

        self.point_cloud_range = [-50.0, -50.0, -5.0, 50.0, 50.0, 3.0]
        if occ_type == 'normal':
            self.occupancy_size = [0.5, 0.5, 0.5]
        elif occ_type == 'fine':
            self.occupancy_size = [0.25, 0.25, 0.25]
        elif occ_type == 'coarse':
            self.occupancy_size = [1.0, 1.0, 1.0]
        self.occ_xdim = int((self.point_cloud_range[3] - self.point_cloud_range[0]) / self.occupancy_size[0])
        self.occ_ydim = int((self.point_cloud_range[4] - self.point_cloud_range[1]) / self.occupancy_size[1])
        self.occ_zdim = int((self.point_cloud_range[5] - self.point_cloud_range[2]) / self.occupancy_size[2])

        self.eval_far = eval_far
        self.eval_near = eval_near
        self.far_distance = far_distance  # define for the far foreground object
        self.near_distance = near_distance
        if eval_far or self.eval_near:
            self.obtain_masked_distanced_voxel()
        self.reset()

    def obtain_masked_distanced_voxel(self):
        index_x  = np.arange(self.occ_xdim)
        index_y  = np.arange(self.occ_ydim)
        index_z  = np.arange(self.occ_zdim)
        z, y, x = np.meshgrid(index_z, index_y, index_x, indexing='ij')
        index_xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        points_x = (index_xyz[:, 0] + 0.5) / self.occ_xdim * (self.point_cloud_range[3] - self.point_cloud_range[0]) + self.point_cloud_range[0]
        points_y = (index_xyz[:, 1] + 0.5) / self.occ_ydim * (self.point_cloud_range[4] - self.point_cloud_range[1]) + self.point_cloud_range[1]
        points_z = (index_xyz[:, 2] + 0.5) / self.occ_zdim * (self.point_cloud_range[5] - self.point_cloud_range[2]) + self.point_cloud_range[2]
        points = np.concatenate([points_x.reshape(-1, 1), points_y.reshape(-1, 1), points_z.reshape(-1, 1)], axis=-1)

        points_distance = np.linalg.norm(points[:, :2], axis=-1)  # 只考虑xy平面的距离
        self.far_voxel_mask = points_distance > self.far_distance
        self.near_voxel_mask = points_distance < self.near_distance

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = np.sum(k)
        correct = np.sum((pred[k] == gt[k]))

        return (
            np.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        mean_IU = np.nanmean(iu)
        mean_IU_no_back = np.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, flow_pred=None, flow_true=None, invalid=None):
        self.count += 1
        mask = y_true != 255
        if invalid is not None:
            mask = mask & (invalid != 255)  # 255 is ignored region
        else:
            mask = mask & (np.ones_like(y_true).astype(np.bool))
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        foreground_tp, foreground_fp, foreground_fn = self.get_foreground_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        self.foreground_completion_tp += foreground_tp
        self.foreground_completion_fp += foreground_fp
        self.foreground_completion_fn += foreground_fn

        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(y_pred, y_true, nonempty=mask, class_num=self.n_classes)
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

        # flow info
        if flow_pred is not None:
            self.get_flow_l2_distance(y_pred, y_true, flow_pred, flow_true, mask)

        # foreground object in the specific postion, for example: distance > 30

        if self.eval_far:
            bs = y_pred.shape[0]
            far_mask = np.repeat(self.far_voxel_mask.reshape(1, -1), bs, axis=0)
            far_mask = far_mask & mask
            far_tp, far_fp, far_fn = self.get_foreground_score_completion(y_pred, y_true, far_mask)
            far_tp_sum, far_fp_sum, far_fn_sum = self.get_score_semantic_and_completion(
                y_pred, y_true, nonempty=far_mask, class_num=self.foreground_obj_num)

            self.far_completion_tp += far_tp
            self.far_completion_fp += far_fp
            self.far_completion_fn += far_fn

            self.far_tps += far_tp_sum
            self.far_fps += far_fp_sum
            self.far_fns += far_fn_sum

        if self.eval_near:
            bs = y_pred.shape[0]
            near_mask = np.repeat(self.near_voxel_mask.reshape(1, -1), bs, axis=0)
            near_mask = near_mask & mask
            near_tp, near_fp, near_fn = self.get_foreground_score_completion(y_pred, y_true, near_mask)
            near_tp_sum, near_fp_sum, near_fn_sum = self.get_score_semantic_and_completion(
                y_pred, y_true, nonempty=near_mask, class_num=self.foreground_obj_num)

            self.near_completion_tp += near_tp
            self.near_completion_fp += near_fp
            self.near_completion_fn += near_fn

            self.near_tps += near_tp_sum
            self.near_fps += near_fp_sum
            self.near_fns += near_fn_sum


    def get_flow_l2_distance(self, y_pred, y_true, flow_pred, flow_true, mask=None):
        bs = y_pred.shape[0]

        for idx in range(bs):
            _y_pred = y_pred[idx]
            _y_true = y_true[idx]
            _flow_pred = flow_pred[idx]
            _flow_true = flow_true[idx]

            if mask is not None:
                mask_idx = mask[idx].reshape(-1)
                gt_foreground_mask = (_y_true < 10)  & (mask_idx == 1)
                pred_foreground_mask = (_y_pred < 10) & (mask_idx == 1)
            else:
                gt_foreground_mask = _y_true < 10
                pred_foreground_mask = _y_pred < 10
            _flow_true = _flow_true[gt_foreground_mask & pred_foreground_mask]  # (n, 2)
            _flow_pred = _flow_pred[gt_foreground_mask & pred_foreground_mask]  # (n, 2)

            _flow_true_magnitude = np.linalg.norm(_flow_true, axis=-1)
            # 根据速度大小分多个区间
            for index in range(len(self.flow_regions)-1):
                vel_min, vel_max = self.flow_regions[index], self.flow_regions[index+1]
                vel_mask = (_flow_true_magnitude >= vel_min) & (_flow_true_magnitude < vel_max)
                vel_true = _flow_true[vel_mask]
                vel_pred = _flow_pred[vel_mask]
                point_num = vel_true.shape[0]
                if point_num:
                    diff = vel_true - vel_pred
                    l2_distance = np.sum(np.linalg.norm(diff, axis=-1))
                    self.flow_l2_distances[index] += l2_distance
                    self.flow_point_nums[index] += point_num

    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )*100.0
        else:
            precision, recall, iou = 0, 0, 0

        if self.foreground_completion_tp != 0:
            foreground_precision = self.foreground_completion_tp / (self.foreground_completion_tp + self.foreground_completion_fp)
            foreground_recall = self.foreground_completion_tp / (self.foreground_completion_tp + self.foreground_completion_fn)
            foreground_iou = self.foreground_completion_tp / (
                self.foreground_completion_tp + self.foreground_completion_fp + self.foreground_completion_fn
            )*100.0
        else:
            foreground_precision, foreground_recall, foreground_iou = 0, 0, 0

        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5) * 100.0

        flow_distances = [None, None, None]  # stationary + moving + all
        if self.flow_point_nums[0] != 0:  # output flow
            self.flow_point_nums = np.append(self.flow_point_nums, np.sum(self.flow_point_nums))
            self.flow_l2_distances = np.append(self.flow_l2_distances, np.sum(self.flow_l2_distances))
            for i in range(len(self.flow_point_nums)):
                flow_distances[i] = self.flow_l2_distances[i]/self.flow_point_nums[i]

        far_metrics = None
        if self.eval_far:
            far_metrics = {}
            if self.far_completion_tp != 0:
                far_precision = self.far_completion_tp / (self.far_completion_tp + self.far_completion_fp)
                far_recall = self.far_completion_tp / (self.far_completion_tp + self.far_completion_fn)
                far_iou = self.far_completion_tp / (self.far_completion_tp + self.far_completion_fp + self.far_completion_fn)*100.0
            else:
                far_precision, far_recall, far_iou = 0, 0, 0
            far_iou_ssc = self.far_tps / (self.far_tps + self.far_fps + self.far_fns + 1e-5) * 100.0
            
            far_metrics['far_miou'] = np.mean(far_iou_ssc)
            far_metrics['far_iou'] = far_iou
            far_metrics['far_precision'] = far_precision
            far_metrics['far_recall'] = far_recall
            far_metrics['far_iou_ssc'] = far_iou_ssc
        
        
        near_metrics = None
        if self.eval_near:
            near_metrics = {}
            near_precision = self.near_completion_tp / (self.near_completion_tp + self.near_completion_fp)
            near_recall = self.near_completion_tp / (self.near_completion_tp + self.near_completion_fn)
            near_iou = self.near_completion_tp / (self.near_completion_tp + self.near_completion_fp + self.near_completion_fn)*100.0
            near_iou_ssc = self.near_tps / (self.near_tps + self.near_fps + self.near_fns + 1e-5) * 100.0
            
            near_metrics['near_miou'] = np.mean(near_iou_ssc)
            near_metrics['near_iou'] = near_iou
            near_metrics['near_precision'] = near_precision
            near_metrics['near_recall'] = near_recall
            near_metrics['near_iou_ssc'] = near_iou_ssc
            
        
        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "foreground_iou": foreground_iou,
            "foreground_precision": foreground_precision,
            "foreground_recall": foreground_recall,
            "iou_ssc": iou_ssc,  # class IOU
            "miou": np.mean(iou_ssc),
            "foreground_miou": np.mean(iou_ssc[:self.foreground_obj_num]),
            "flow_distance": flow_distances,
            'far_metrics': far_metrics,
            'near_metrics': near_metrics,
        }

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0

        self.foreground_completion_tp = 0
        self.foreground_completion_fp = 0
        self.foreground_completion_fn = 0

        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)
        
        
        # far foreground object
        self.far_completion_tp = 0 
        self.far_completion_fp = 0 
        self.far_completion_fn = 0 

        self.far_tps = np.zeros(self.foreground_obj_num)
        self.far_fps = np.zeros(self.foreground_obj_num)
        self.far_fns = np.zeros(self.foreground_obj_num)
        
        
        # near foreground object
        self.near_completion_tp = 0 
        self.near_completion_fp = 0 
        self.near_completion_fn = 0 

        self.near_tps = np.zeros(self.foreground_obj_num)
        self.near_fps = np.zeros(self.foreground_obj_num)
        self.near_fns = np.zeros(self.foreground_obj_num)
        

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)

        # flow: 使用速度阈值0.2来区分运动或者静止的flow gt
        self.flow_regions = [-0.2, 0.2, 100]  # stationary + mov
        self.flow_point_nums = np.zeros(len(self.flow_regions)-1).astype(np.int64)  # 2 state
        self.flow_l2_distances = np.zeros(len(self.flow_regions)-1)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size

        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 640000)
        predict = predict.reshape(_bs, -1)  # (_bs, 640000), 16*200*200 = 640000
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict != self.empty_label] = 1
        b_true[target != self.empty_label] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum


    def get_foreground_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size

        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 640000)
        predict = predict.reshape(_bs, -1)  # (_bs, 640000), 16*200*200 = 640000
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)

        b_pred[predict != self.empty_label] = 1
        b_true[target != self.empty_label] = 1

        b_pred[predict >= self.foreground_obj_num] = 0
        b_true[target >= self.foreground_obj_num] = 0

        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum


    def get_score_semantic_and_completion(self, predict, target, nonempty=None, class_num=16):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]  # batch size

        target = target.reshape(_bs, -1)  # (_bs, 640000)
        predict = predict.reshape(_bs, -1)  # (_bs, 640000), 16*200*200 = 640000

        cnt_class = np.zeros(class_num, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(class_num, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(class_num, dtype=np.int32)  # tp
        fp_sum = np.zeros(class_num, dtype=np.int32)  # fp
        fn_sum = np.zeros(class_num, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[nonempty_idx == 1]
                y_true = y_true[nonempty_idx == 1]
            for j in range(class_num):  # for each class
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum
