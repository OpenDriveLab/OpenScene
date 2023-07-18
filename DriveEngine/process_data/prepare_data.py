import os, mmcv

config = {
    'root': '<your_path_to_openscene>/openscene-v1.1/meta_datas'
}

for data_type in ['mini']: # train/val/test/mini
    print('process scenes:', data_type)
    scenes_dir = os.path.join(config['root'], f'meta_data_{data_type}.pkl')
    if not os.path.exists(scenes_dir):
        print("skip {} data".format(data_type))
        continue

    data = mmcv.load(scenes_dir)
    for cur_info in data['infos']:
        occ_gt_final_path = cur_info['occ_gt_final_path']
        flow_gt_final_path = cur_info['flow_gt_final_path']
        cur_info['occ_gt_final_path'] = os.path.join(os.getcwd(), occ_gt_final_path)
        cur_info['flow_gt_final_path'] = os.path.join(os.getcwd(), flow_gt_final_path)

        cams = cur_info['cams']

        cam_types = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_B0', 'CAM_R2']
        for cam_type in cam_types:
            data_path = cams[cam_type]['data_path']
            cams[cam_type]['data_path'] = os.path.join(os.getcwd(), data_path)

        # pdb.set_trace()

    save_path = os.path.join('meta_data_{}.pkl'.format(data_type))
    mmcv.dump(data, save_path)
