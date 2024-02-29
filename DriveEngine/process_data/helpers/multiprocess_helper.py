import numpy as np
import multiprocessing


def get_scenes_per_thread(scenes, thread_num):
    scenes = sorted(scenes)
    num_tasks = thread_num
    cur_id = int(multiprocessing.current_process().name)

    num_scene = len(scenes)
    a = num_scene//num_tasks
    b = num_scene % num_tasks

    if cur_id == 0:
        print('num_scene:', num_scene)

    process_num = []
    for id in range(num_tasks):
        if id >= b:
            process_num.append(a)
        else:
            process_num.append(a+1)
    addsum = np.cumsum(process_num)

    if cur_id == 0:
        start = 0
        end = addsum[0]
    else:
        start = addsum[cur_id-1]
        end = addsum[cur_id]

    return scenes[start:end], start
