from __future__ import print_function, division

import os
import csv
import time
import copy

import torch
import numpy as np
import torchvision
import scipy.io as sio

from evaluate import fx_calc_map_multilabel

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def calc_label_sim(label_1, label_2):
    return label_1.float().mm(label_2.float().t())


def _compute_valid_map(model, valid_loader, device):
    """Compute validation I2T/T2I mAP with signed hash codes."""
    t_imgs, t_txts, t_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, txts, labels, _, _ in valid_loader:
            imgs = imgs.to(device, non_blocking=True)
            txts = txts.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            v1, v2 = model(imgs, txts)
            t_imgs.append(v1.sign().cpu().numpy())
            t_txts.append(v2.sign().cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    if len(t_imgs) == 0:
        return 0.0, 0.0

    t_imgs = np.concatenate(t_imgs, axis=0)
    t_txts = np.concatenate(t_txts, axis=0)
    t_labels = np.concatenate(t_labels, axis=0)
    img2txt = fx_calc_map_multilabel(t_imgs, t_txts, t_labels, metric="hamming")
    txt2img = fx_calc_map_multilabel(t_txts, t_imgs, t_labels, metric="hamming")
    return float(img2txt), float(txt2img)


def _prepare_valid_curve_paths(configs):
    log_dir = os.path.join(".", "outputs", "valid_curves")
    os.makedirs(log_dir, exist_ok=True)
    ds = configs.dataset
    nr = configs.noisy_ratio
    nm = configs.noise_mode
    sd = configs.seed
    bt = configs.bit
    lt = configs.loss_type
    tag = f"{ds}_nr{nr}_{nm}_seed{sd}_{lt}_{bt}bit"
    csv_path = os.path.join(log_dir, f"valid_curve_{tag}.csv")
    mat_path = os.path.join(log_dir, f"valid_curve_{tag}.mat")
    npy_path = os.path.join(log_dir, f"valid_curve_{tag}.npy")
    return csv_path, mat_path, npy_path


def _append_valid_curve(csv_path, epoch, img2txt, txt2img, avg_map):
    new_file = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["epoch", "valid_i2t", "valid_t2i", "valid_avg"])
        w.writerow([int(epoch), float(img2txt), float(txt2img), float(avg_map)])


def _save_valid_curve_snapshots(mat_path, npy_path, epochs, i2t_list, t2i_list, avg_list):
    epochs_np = np.asarray(epochs, dtype=np.int32)
    i2t_np = np.asarray(i2t_list, dtype=np.float32)
    t2i_np = np.asarray(t2i_list, dtype=np.float32)
    avg_np = np.asarray(avg_list, dtype=np.float32)
    sio.savemat(
        mat_path,
        {"epoch": epochs_np, "valid_i2t": i2t_np, "valid_t2i": t2i_np, "valid_avg": avg_np},
    )
    np.save(
        npy_path,
        {"epoch": epochs_np, "valid_i2t": i2t_np, "valid_t2i": t2i_np, "valid_avg": avg_np},
        allow_pickle=True,
    )


def _parse_int_set_env(name, default_csv):
    s = os.environ.get(name, default_csv)
    out = set()
    for x in str(s).split(","):
        x = x.strip()
        if x == "":
            continue
        try:
            out.add(int(x))
        except Exception:
            pass
    return out


def _get_run_dir(configs):
    tag = f"{configs.dataset}_nr{configs.noisy_ratio}_{configs.noise_mode}_seed{configs.seed}_{configs.loss_type}_{configs.bit}bit"
    return os.path.join(".", "outputs", "runs", tag)


def _dump_epoch_weight_logs(run_dir, epoch, weight_all, clean_mask_all, noisy_mask_all, img2txt, txt2img):
    out_dir = os.path.join(run_dir, "weight_logs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"train_weights_epoch_{epoch}.npz")
    payload = {
        "epoch": np.array([int(epoch)], dtype=np.int32),
        "weight": weight_all.astype(np.float32),
        "clean_mask": clean_mask_all.astype(np.uint8),
        "noisy_mask": noisy_mask_all.astype(np.uint8),
        "valid_i2t": np.array([float(img2txt)], dtype=np.float32),
        "valid_t2i": np.array([float(txt2img)], dtype=np.float32),
        "valid_avg": np.array([float((img2txt + txt2img) / 2.0)], dtype=np.float32),
    }
    np.savez(out, **payload)
    print(f"[WEIGHT_LOG] saved {out}  active_rate={(weight_all > 0).mean():.4f}")


def train_model(model, data_loaders, criterion, optimizer, configs, scheduler=None):
    device = next(model.parameters()).device
    num_epochs = configs.MAX_EPOCH

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    valid_csv_path, valid_mat_path, valid_npy_path = _prepare_valid_curve_paths(configs)
    print(f"[VALID_CURVE] CSV: {valid_csv_path}")
    print(f"[VALID_CURVE] MAT: {valid_mat_path}")
    print(f"[VALID_CURVE] NPY: {valid_npy_path}")

    curve_epochs, curve_i2t, curve_t2i, curve_avg = [], [], [], []
    run_dir = _get_run_dir(configs)
    os.makedirs(run_dir, exist_ok=True)
    weight_log_epochs = _parse_int_set_env("WEIGHT_LOG_EPOCHS", "5,99")

    train_ds = data_loaders["train"].dataset
    N_train = len(train_ds)
    MAP_list = []

    for epoch in range(num_epochs):
        ep = epoch + 1
        print("Epoch {}/{}".format(ep, num_epochs))
        print("-" * 20)

        epoch_weight_all = np.zeros((N_train,), dtype=np.float32)
        epoch_clean_mask_all = np.zeros((N_train,), dtype=np.uint8)
        epoch_noisy_mask_all = np.zeros((N_train,), dtype=np.uint8)

        train_batches = 0
        train_Lfuzzy_sum = 0.0
        train_Lfar_sum = 0.0
        train_Lintra_sum = 0.0
        train_Lhash_sum = 0.0
        epoch_loss_train = None
        epoch_loss_valid = None

        cbar_warmup_min, cbar_warmup_max, cbar_warmup_sum, cbar_warmup_cnt = 1e9, -1e9, 0.0, 0
        cbar_finetune_min, cbar_finetune_max, cbar_finetune_sum, cbar_finetune_cnt = 1e9, -1e9, 0.0, 0

        for phase in ["train", "valid"]:
            model.train(mode=(phase == "train"))
            running_loss = 0.0

            for imgs, txts, labels, ori_labels, index in data_loaders[phase]:
                imgs = imgs.to(device, non_blocking=True)
                txts = txts.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                ori_labels = ori_labels.to(device, non_blocking=True)
                index = index.to(device, non_blocking=True)

                if torch.isnan(imgs).any() or torch.isnan(txts).any():
                    print("Data contains NaN.")
                    continue

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    v1, v2 = model(imgs, txts)
                    m1 = model.membership(v1)
                    m2 = model.membership(v2)
                    result = criterion(v1, v2, m1, m2, labels, epoch, configs)

                    if isinstance(result, tuple) and len(result) == 2:
                        loss, info = result
                        if isinstance(info, dict):
                            weight = info.get("weights", torch.ones(imgs.size(0), device=device))
                            if phase == "train":
                                train_batches += 1
                                train_Lfuzzy_sum += float(info.get("L_fuzzy", 0.0))
                                train_Lfar_sum += float(info.get("L_FAR", 0.0))
                                train_Lintra_sum += float(info.get("L_intra", 0.0))
                                train_Lhash_sum += float(info.get("L_hash", 0.0))
                                stage = str(info.get("stage", "")).strip()

                                if all(k in info for k in ["cbar_warmup_min", "cbar_warmup_max", "cbar_warmup_mean"]):
                                    if stage == "Warm-up":
                                        cb_min = float(info["cbar_warmup_min"])
                                        cb_max = float(info["cbar_warmup_max"])
                                        cb_mean = float(info["cbar_warmup_mean"])
                                        cbar_warmup_min = min(cbar_warmup_min, cb_min)
                                        cbar_warmup_max = max(cbar_warmup_max, cb_max)
                                        cbar_warmup_sum += cb_mean
                                        cbar_warmup_cnt += 1

                                if all(k in info for k in ["cbar_finetune_min", "cbar_finetune_max", "cbar_finetune_mean"]):
                                    if stage == "Fine-tuning":
                                        cb_min = float(info["cbar_finetune_min"])
                                        cb_max = float(info["cbar_finetune_max"])
                                        cb_mean = float(info["cbar_finetune_mean"])
                                        cbar_finetune_min = min(cbar_finetune_min, cb_min)
                                        cbar_finetune_max = max(cbar_finetune_max, cb_max)
                                        cbar_finetune_sum += cb_mean
                                        cbar_finetune_cnt += 1
                        else:
                            weight = info
                            if phase == "train":
                                train_batches += 1
                    else:
                        loss = result
                        weight = torch.ones(imgs.size(0), device=device)
                        if phase == "train":
                            train_batches += 1

                    if phase == "train":
                        idx_np = index.detach().cpu().numpy()
                        w_np = weight.detach().float().cpu().numpy()
                        epoch_weight_all[idx_np] = w_np
                        cm = (torch.argmax(labels, dim=1) == torch.argmax(ori_labels, dim=1)).detach().cpu().numpy().astype(np.uint8)
                        epoch_clean_mask_all[idx_np] = cm
                        epoch_noisy_mask_all[idx_np] = (1 - cm)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()

                running_loss += float(loss.item())

            epoch_loss = running_loss / max(1, len(data_loaders[phase].dataset))
            if phase == "train":
                epoch_loss_train = epoch_loss
            else:
                epoch_loss_valid = epoch_loss

        img2txt, txt2img = _compute_valid_map(model, data_loaders["valid"], device)
        avg_map = (img2txt + txt2img) / 2.0
        MAP_list.append(avg_map)
        _append_valid_curve(valid_csv_path, ep, img2txt, txt2img, avg_map)
        curve_epochs.append(ep)
        curve_i2t.append(img2txt)
        curve_t2i.append(txt2img)
        curve_avg.append(avg_map)
        _save_valid_curve_snapshots(valid_mat_path, valid_npy_path, curve_epochs, curve_i2t, curve_t2i, curve_avg)

        cur_lr = optimizer.param_groups[0]["lr"]
        print("{:5s}  Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}  lr: {:g}".format(
            "train", float(epoch_loss_train if epoch_loss_train is not None else 0.0), img2txt, txt2img, cur_lr
        ))
        print("{:5s}  Loss: {:.4f} Img2Txt: {:.4f}  Txt2Img: {:.4f}  lr: {:g}".format(
            "valid", float(epoch_loss_valid if epoch_loss_valid is not None else 0.0), img2txt, txt2img, cur_lr
        ))

        if avg_map > best_acc:
            best_acc = avg_map
            best_model_wts = copy.deepcopy(model.state_dict())

        if train_batches > 0:
            print(
                f"[LOSS] Epoch {ep}/{num_epochs}  "
                f"L_fuzzy={train_Lfuzzy_sum / train_batches:.6f}  "
                f"L_FAR={train_Lfar_sum / train_batches:.6f}  "
                f"L_intra={train_Lintra_sum / train_batches:.6f}  "
                f"L_hash={train_Lhash_sum / train_batches:.6f}"
            )
        else:
            print(f"[LOSS] Epoch {ep}/{num_epochs}  L_fuzzy=N/A  L_FAR=N/A  L_intra=N/A  L_hash=N/A")

        if cbar_warmup_cnt > 0:
            print("[CBAR] Epoch {:3d} | Warm-up c_bar range: [{:.6f}, {:.6f}]  mean={:.6f}".format(
                ep, float(cbar_warmup_min), float(cbar_warmup_max), float(cbar_warmup_sum / max(1, cbar_warmup_cnt))
            ))
        if cbar_finetune_cnt > 0:
            print("[CBAR] Epoch {:3d} | Fine-tuning c_bar range: [{:.6f}, {:.6f}]  mean={:.6f}".format(
                ep, float(cbar_finetune_min), float(cbar_finetune_max), float(cbar_finetune_sum / max(1, cbar_finetune_cnt))
            ))

        if ep in weight_log_epochs:
            _dump_epoch_weight_logs(
                run_dir=run_dir,
                epoch=ep,
                weight_all=epoch_weight_all,
                clean_mask_all=epoch_clean_mask_all,
                noisy_mask_all=epoch_noisy_mask_all,
                img2txt=img2txt,
                txt2img=txt2img,
            )

        if scheduler is not None:
            try:
                scheduler.step()
            except TypeError:
                scheduler.step(epoch)
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"[LR Scheduler] Epoch {ep}/{num_epochs}: lr = {cur_lr:.8e}")

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best average ACC: {:4f}".format(best_acc))
    model.load_state_dict(best_model_wts)
    return model, MAP_list
