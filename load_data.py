from torch.utils.data.dataset import Dataset
import scipy.io as sio
from torch.utils.data import DataLoader
import numpy as np
import os


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labels, ori_labels):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.ori_labels = ori_labels
        self.soft_labels = self.labels.astype(np.float32).copy()

    def __getitem__(self, index):
        return (
            self.images[index],
            self.texts[index],
            self.labels[index],
            self.ori_labels[index],
            index,
        )

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def update_soft_labels_by_indices(self, indices, new_soft):
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        if hasattr(new_soft, "detach"):
            new_soft = new_soft.detach().cpu().numpy()
        self.soft_labels[indices] = new_soft.astype(np.float32, copy=False)

    def update_hard_labels_by_indices(self, indices, new_onehot):
        if hasattr(indices, "detach"):
            indices = indices.detach().cpu().numpy()
        if hasattr(new_onehot, "detach"):
            new_onehot = new_onehot.detach().cpu().numpy()
        self.labels[indices] = new_onehot.astype(self.labels.dtype, copy=False)

    def update_labels_by_indices(self, indices, new_onehot):
        self.update_hard_labels_by_indices(indices, new_onehot)


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def get_noisylabels(labels, noisy_radio, noise_mode, seed=10, use_tqdm=False, asym_map=None):
    labels = np.asarray(labels)
    data_num, class_num = labels.shape

    rng = np.random.default_rng(seed)
    num_noise = int(data_num * noisy_radio)
    if num_noise <= 0:
        return labels.copy()

    noise_indices = rng.choice(data_num, size=num_noise, replace=False)
    noisy = labels.copy()

    if noise_mode == 'sym':
        it = noise_indices
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(it)

        for i in it:
            ones_idx = np.flatnonzero(noisy[i] > 0)
            zeros_idx = np.flatnonzero(noisy[i] == 0)
            if ones_idx.size > 0:
                noisy[i, int(rng.choice(ones_idx))] = 0
            if zeros_idx.size > 0:
                noisy[i, int(rng.choice(zeros_idx))] = 1

    elif noise_mode == 'asym':
        if class_num < 2:
            return noisy

        if asym_map is None:
            class_map = (np.arange(class_num, dtype=np.int64) + 1) % class_num
        elif isinstance(asym_map, dict):
            class_map = np.arange(class_num, dtype=np.int64)
            for k, v in asym_map.items():
                class_map[int(k)] = int(v)
        else:
            class_map = np.asarray(asym_map, dtype=np.int64).reshape(-1)
            if class_map.size != class_num:
                raise ValueError(f"asym_map length must be {class_num}, got {class_map.size}")

        if np.any(class_map < 0) or np.any(class_map >= class_num):
            raise ValueError("asym_map contains out-of-range class index")

        it = noise_indices
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(it)

        for i in it:
            ones_idx = np.flatnonzero(noisy[i] > 0)
            if ones_idx.size == 0:
                continue
            src = int(rng.choice(ones_idx))
            tgt = int(class_map[src])
            if tgt == src or noisy[i, tgt] > 0:
                cur = tgt
                found = False
                for _ in range(class_num - 1):
                    cur = int(class_map[cur])
                    if cur != src and noisy[i, cur] == 0:
                        tgt = cur
                        found = True
                        break
                if not found:
                    continue
            noisy[i, src] = 0
            noisy[i, tgt] = 1
    else:
        raise ValueError(f"Unknown noise_mode: {noise_mode}")

    return noisy


def get_loader(data_name, batch_size, noisy_ratio, noise_mode, data_path=None):
    np.random.seed(1)

    if data_name == 'wiki':
        valid_len = 231
        data = sio.loadmat('../datasets/wiki.mat')
        img_train = data['train_imgs_deep']
        text_train = data['train_texts_doc']
        label_train_img = data['train_imgs_labels'].reshape([-1, 1]).astype('int16')
        img_test = data['test_imgs_deep']
        text_test = data['test_texts_doc']
        label_test_img = data['test_imgs_labels'].reshape([-1, 1]).astype('int16')
        img_valid = img_test[0:valid_len]
        text_valid = text_test[0:valid_len]
        label_valid_img = label_test_img[0:valid_len]
        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test_img = label_test_img[valid_len:]

    elif data_name == 'xmedia':
        valid_len = 500
        all_data = sio.loadmat('/ICML/xinxin37/datasets/XMediaFeatures.mat')
        img_test = all_data['I_te_CNN'].astype('float32')
        img_train = all_data['I_tr_CNN'].astype('float32')
        text_test = all_data['T_te_BOW'].astype('float32')
        text_train = all_data['T_tr_BOW'].astype('float32')
        label_test_img = all_data['teImgCat'].reshape([-1, 1]).astype('int64')
        label_train_img = all_data['trImgCat'].reshape([-1, 1]).astype('int64')
        img_valid = img_test[0:valid_len]
        text_valid = text_test[0:valid_len]
        label_valid_img = label_test_img[0:valid_len]
        img_test = img_test[valid_len:]
        text_test = text_test[valid_len:]
        label_test_img = label_test_img[valid_len:]

    elif data_name == 'INRIA-Websearch':
        data = sio.loadmat(data_path or '../datasets/INRIA-Websearch.mat')
        if 'tr_img' in data:
            img_train = data['tr_img'].astype('float32')
            text_train = data['tr_txt'].astype('float32')
            label_train_img = data['tr_img_lab'].reshape([-1, 1]).astype('int16')
            img_valid = data['val_img'].astype('float32')
            text_valid = data['val_txt'].astype('float32')
            label_valid_img = data['val_img_lab'].reshape([-1, 1]).astype('int16')
            img_test = data['te_img'].astype('float32')
            text_test = data['te_txt'].astype('float32')
            label_test_img = data['te_img_lab'].reshape([-1, 1]).astype('int16')
        else:
            img_train = data['img_train'].astype('float32')
            text_train = data['text_train'].astype('float32')
            label_train_img = data['label_train'].reshape([-1, 1]).astype('int16')
            img_valid = data['img_valid'].astype('float32')
            text_valid = data['text_valid'].astype('float32')
            label_valid_img = data['label_valid'].reshape([-1, 1]).astype('int16')
            img_test = data['img_test'].astype('float32')
            text_test = data['text_test'].astype('float32')
            label_test_img = data['label_test'].reshape([-1, 1]).astype('int16')

    elif data_name == 'xmedianet':
        all_data = sio.loadmat('../datasets/XMediaNet5View_Doc2Vec.mat')
        img_train = all_data['img_train'].astype('float32')
        img_valid = all_data['img_valid'].astype('float32')
        img_test = all_data['img_test'].astype('float32')
        text_train = all_data['text_train'].astype('float32')
        text_valid = all_data['text_valid'].astype('float32')
        text_test = all_data['text_test'].astype('float32')
        label_train_img = all_data['label_train'].reshape([-1, 1]).astype('int64', copy=False)
        label_valid_img = all_data['label_valid'].reshape([-1, 1]).astype('int64', copy=False)
        label_test_img = all_data['label_test'].reshape([-1, 1]).astype('int64', copy=False)

    elif data_name == 'iapr-tc12':
        print("Loading IAPR-TC12 multilabel dataset (train = Database, test = Test)...")
        data = sio.loadmat('/ICML/xinxin37/datasets/iapr-tc12-rand.mat')
        img_train = data['VDatabase'].astype('float32')
        text_train = data['YDatabase'].astype('float32')
        label_train_img = data['databaseL'].astype('int16')
        img_test = data['VTest'].astype('float32')
        text_test = data['YTest'].astype('float32')
        label_test_img = data['testL'].astype('int16')
        img_valid = img_test
        text_valid = text_test
        label_valid_img = label_test_img
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    img_train = img_train.astype('float32')
    img_valid = img_valid.astype('float32')
    img_test = img_test.astype('float32')
    text_train = text_train.astype('float32')
    text_valid = text_valid.astype('float32')
    text_test = text_test.astype('float32')

    label_train = label_train_img
    label_valid = label_valid_img
    label_test = label_test_img
    if len(label_train.shape) == 1 or label_train.shape[1] == 1:
        label_train = ind2vec(label_train.reshape([-1, 1])).astype('int16')
        label_valid = ind2vec(label_valid.reshape([-1, 1])).astype('int16')
        label_test = ind2vec(label_test.reshape([-1, 1])).astype('int16')

    print('train shape: ', img_train.shape[0], 'valid shape:', img_valid.shape[0], 'test shape:', img_test.shape[0])

    root_dir = 'noisy_labels'
    os.makedirs(root_dir, exist_ok=True)
    noise_file = os.path.join(root_dir, f'{data_name}_noise_labels_{noisy_ratio:g}_{noise_mode}.mat')
    label_noisy = get_noisylabels(label_train, noisy_ratio, noise_mode)
    sio.savemat(noise_file, {'noisy_label': label_noisy})

    imgs = {'train': img_train, 'valid': img_valid}
    texts = {'train': text_train, 'valid': text_valid}
    labels = {'train': label_noisy, 'valid': label_valid}
    ori_labels = {'train': label_train, 'valid': label_valid}
    dataset = {
        x: CustomDataSet(images=imgs[x], texts=texts[x], labels=labels[x], ori_labels=ori_labels[x])
        for x in ['train', 'valid']
    }
    shuffle = {'train': True, 'valid': False}
    dataloader = {
        x: DataLoader(dataset[x], batch_size=batch_size, shuffle=shuffle[x], num_workers=0)
        for x in ['train', 'valid']
    }

    input_data_par = {
        'img_test': img_test,
        'text_test': text_test,
        'label_test': label_test,
        'img_dim': img_train.shape[1],
        'text_dim': text_train.shape[1],
        'num_train': img_train.shape[0],
        'num_class': label_train.shape[1],
    }
    return dataloader, input_data_par
