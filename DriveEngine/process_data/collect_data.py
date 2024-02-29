import pickle
import os

if __name__ == '__main__':
    mini_infos = os.listdir('dataset/openscene-v1.1/meta_datas/mini')
    mini_infos = [os.path.join('dataset/openscene-v1.1/meta_datas/mini', each) for each in mini_infos if each.endswith('.pkl')]

    train_paths = mini_infos[:int(len(mini_infos) * 0.85)]
    val_paths = mini_infos[int(len(mini_infos) * 0.85):]

    train_infos = []
    for file in train_paths:
        with open(file, 'rb') as f:
            train_infos.extend(pickle.load(f))

    val_infos = []
    for file in val_paths:
        with open(file, 'rb') as f:
            val_infos.extend(pickle.load(f))

    with open('data/openscene_mini_train.pkl', 'wb') as f:
        pickle.dump(train_infos, f)
    
    with open('data/openscene_mini_val.pkl', 'wb') as f:
        pickle.dump(val_infos, f)
