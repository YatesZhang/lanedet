import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import subprocess
import sys
from shutil import rmtree
import cv2
import numpy as np
from config.culane import culane_evaluator_path, env

def check():
    FNULL = open(os.devnull, 'w')
    result = subprocess.call(
        culane_evaluator_path, stdout=FNULL, stderr=FNULL)
    if result > 1:
        print('There is something wrong with evaluate tool, please compile it.')
        sys.exit()


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res


def call_culane_eval(data_dir, output_path='./output'):
    print("call culane eval: output_path:" + output_path)
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = output_path + '/'
    print("detect_dir: " + detect_dir)
    # print("call culane eval: ")
    w_lane = 30
    iou = 0.5;  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir, 'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir, 'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir, 'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir, 'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir, 'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir, 'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir, 'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir, 'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out0 = os.path.join(output_path, 'txt', 'out0_normal.txt')
    out1 = os.path.join(output_path, 'txt', 'out1_crowd.txt')
    out2 = os.path.join(output_path, 'txt', 'out2_hlight.txt')
    out3 = os.path.join(output_path, 'txt', 'out3_shadow.txt')
    out4 = os.path.join(output_path, 'txt', 'out4_noline.txt')
    out5 = os.path.join(output_path, 'txt', 'out5_arrow.txt')
    out6 = os.path.join(output_path, 'txt', 'out6_curve.txt')
    out7 = os.path.join(output_path, 'txt', 'out7_cross.txt')
    out8 = os.path.join(output_path, 'txt', 'out8_night.txt')

    eval_cmd = culane_evaluator_path

    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list1, w_lane, iou, im_w, im_h, frame, out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list2, w_lane, iou, im_w, im_h, frame, out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list3, w_lane, iou, im_w, im_h, frame, out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list4, w_lane, iou, im_w, im_h, frame, out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list5, w_lane, iou, im_w, im_h, frame, out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list6, w_lane, iou, im_w, im_h, frame, out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list7, w_lane, iou, im_w, im_h, frame, out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'
              % (eval_cmd, data_dir, detect_dir, data_dir, list8, w_lane, iou, im_w, im_h, frame, out8))
    res_all = {}
    res_all['normal'] = read_helper(out0)
    res_all['crowd'] = read_helper(out1)
    res_all['night'] = read_helper(out8)
    res_all['noline'] = read_helper(out4)
    res_all['shadow'] = read_helper(out3)
    res_all['arrow'] = read_helper(out5)
    res_all['hlight'] = read_helper(out2)
    res_all['curve'] = read_helper(out6)
    res_all['cross'] = read_helper(out7)
    return res_all


def probmap2lane(probmaps, exists, pts=18):
    coords = []
    if probmaps.shape[1]>=4:
        probmaps = probmaps[1:, ...]    # no background
    exists = exists > 0.5
    for probmap, exist in zip(probmaps, exists):
        if exist == 0:
            continue
        probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
        thr = 0.3
        coordinate = np.zeros(pts)
        cut_height = env['cut_height']
        img_height = 288
        img_width = 800
        ori_imgh = 590
        ori_imgw = 1640
        for i in range(pts):
            line = probmap[round(
                img_height - i * 20 / (ori_imgh - cut_height) * img_height) - 1]

            if np.max(line) > thr:
                coordinate[i] = np.argmax(line) + 1
        if np.sum(coordinate > 0) < 2:
            continue

        img_coord = np.zeros((pts, 2))
        img_coord[:, :] = -1
        for idx, value in enumerate(coordinate):
            if value > 0:
                img_coord[idx][0] = round(value * ori_imgw / img_width - 1)
                img_coord[idx][1] = round(ori_imgh - idx * 20 - 1)

        img_coord = img_coord.astype(int)
        coords.append(img_coord)

    return coords


class CULaneEval(nn.Module):
    def __init__(self, cfg):
        super(CULaneEval, self).__init__()

        # Firstly, check the evaluation tool
        check()
        self.cfg = cfg
        self.logger = logging.getLogger(self.cfg.log_name)
        
        # the runner will set the out_dir: os.path.join(work_dir, 'lines')
        self.out_dir = None
        if cfg.view:
            self.view_dir = os.path.join(self.cfg.work_dir, 'vis')

    def evaluate(self, output, batch_path, with_heat_map=False):
        seg, exists = output['seg'], output['exist']
        seg = F.softmax(seg, dim=1).cpu().numpy()

        # predictmaps: 
        predictmaps = seg

        exists = exists.cpu().numpy()
        batch_size = seg.shape[0]
        img_name = batch_path
        for i in range(batch_size):
            coords = probmap2lane(predictmaps[i], exists[i])
            outname = self.out_dir + img_name[i][:-4] + '.lines.txt'
            outdir = os.path.dirname(outname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            f = open(outname, 'w')
            for coord in coords:
                for x, y in coord:
                    if x < 0 and y < 0:
                        continue
                    f.write('%d %d ' % (x, y))
                f.write('\n')
            f.close()

    def summarize(self):
        # eval_list_path = self.cfg.test_txt
        self.logger.info('summarize result ... ')
        # prob2lines(self.prob_dir, self.out_dir, eval_list_path, self.cfg)
        res = call_culane_eval(self.cfg.data_root, output_path=self.out_dir)
        TP, FP, FN = 0, 0, 0
        out_str = 'Copypaste: '
        for k, v in res.items():
            val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
            val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
            val_p, val_r, val_f1 = float(v['precision']), float(v['recall']), float(v['Fmeasure'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            self.logger.info(k + ': ' + str(v))
            out_str += k
            for metric, value in v.items():
                out_str += ' ' + str(value).rstrip('\n')
            out_str += ' '
        P = TP * 1.0 / (TP + FP + 1e-9)
        R = TP * 1.0 / (TP + FN + 1e-9)
        F = 2*P*R/(P + R + 1e-9)
        overall_result_str = ('Overall Precision: %f Recall: %f F1: %f' % (P, R, F))
        self.logger.info(overall_result_str)
        out_str = out_str + overall_result_str
        self.logger.info(out_str)

        # delete the tmp output
        # rmtree(self.out_dir)

        # return F1 measures:
        return F



