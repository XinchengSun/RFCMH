
import torch
import torch.nn as nn
import torch.nn.functional as F


def intra_loss(fea, tau=1.0, q=1.0):
    n_view = 2
    batch_size = fea[0].shape[0]

    all_fea = torch.cat(fea, dim=0)  # [2B, D]
    sim = all_fea.mm(all_fea.t())  # [2B, 2B]
    sim = torch.exp(sim / tau)
    sim = sim.clone()
    sim.fill_diagonal_(0.0)

    sim_sum1 = sum([sim[:, v * batch_size:(v + 1) * batch_size] for v in range(n_view)])
    diag1 = torch.cat(
        [sim_sum1[v * batch_size:(v + 1) * batch_size].diag() for v in range(n_view)]
    )
    p1 = diag1 / (sim.sum(1) + 1e-8)
    loss1 = (1 - q) * (1. - p1.clamp_min(1e-8) ** q).div(max(q, 1e-8)) + q * (1 - p1)

    sim_sum2 = sum([sim[v * batch_size:(v + 1) * batch_size] for v in range(n_view)])
    diag2 = torch.cat(
        [sim_sum2[:, v * batch_size:(v + 1) * batch_size].diag() for v in range(n_view)]
    )
    p2 = diag2 / (sim.sum(1) + 1e-8)
    loss2 = (1 - q) * (1. - p2.clamp_min(1e-8) ** q).div(max(q, 1e-8)) + q * (1 - p2)

    return loss1.mean() + loss2.mean()


def get_train_category_credibility(predict, labels, eps: float = 1e-6, comp_topk: int = 4):
    predict = predict.clamp(0.0, 1.0)
    labels = labels.float()

    B, K = predict.shape
    k = int(comp_topk)

    if k <= 0:
        w_poss = 1.0
        w_nec = 0.0
        k_eff = 1
    else:
        w_nec = float(k) / float(k + 1)
        w_poss = 1.0 / float(k + 1)
        k_eff = min(k, K)

    label_mask = labels > 0.5
    label_prob = (predict * labels).sum(dim=1, keepdim=True)  # [B,1]
    non_label_predict = predict * (1.0 - labels)  # [B,K]

    if k_eff <= 1:
        comp = non_label_predict.max(dim=1, keepdim=True)[0]  # [B,1]
    else:
        topk_vals = non_label_predict.topk(
            k=k_eff, dim=1, largest=True, sorted=False
        ).values  # [B,k_eff]
        comp = topk_vals.mean(dim=1, keepdim=True)  # [B,1]

    necessity = torch.where(
        label_mask,
        1.0 - comp.expand_as(predict),          # label classes
        1.0 - label_prob.expand_as(predict)     # non-label classes
    )

    credibility = w_poss * predict + w_nec * necessity
    return credibility.clamp(eps, 1.0 - eps)


class FARLoss(nn.Module):
    def __init__(self, num_classes=100,
                 comp_topk: int = 1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.comp_topk = int(comp_topk)

    def _normalize_rows(self, x: torch.Tensor, eps: float = 1e-12):
        return x / (x.sum(dim=1, keepdim=True) + eps)

    def _sharpen_probs(self, p: torch.Tensor,
                       gamma: float = 2.0,
                       eps: float = 1e-12):
        p = p.clamp(eps, 1.0 - eps)
        q = p ** gamma
        q = q / (q.sum(dim=1, keepdim=True) + eps)
        return q

    def _warmup_fuzzy_loss(self, v1_pred, v2_pred, labels):
        p1 = self._normalize_rows(v1_pred)
        p2 = self._normalize_rows(v2_pred)

        gamma_f = 1.0
        if gamma_f > 1.0:
            p1 = self._sharpen_probs(p1, gamma=gamma_f)
            p2 = self._sharpen_probs(p2, gamma=gamma_f)

        cred1 = get_train_category_credibility(p1, labels, comp_topk=self.comp_topk)  # [B,K]
        cred2 = get_train_category_credibility(p2, labels, comp_topk=self.comp_topk)  # [B,K]
        c_bar = 0.5 * (cred1 + cred2)

        y = labels.float()
        loss_per_sample = (
            (cred1 - y).pow(2).sum(dim=1) +
            (cred2 - y).pow(2).sum(dim=1)
        )  # [B]

        weights = torch.ones_like(loss_per_sample)
        L_fuzzy = (weights * loss_per_sample).sum() / (weights.sum() + 1e-8)

        stats = {
            "cbar_warmup_min": float(c_bar.min().item()),
            "cbar_warmup_max": float(c_bar.max().item()),
            "cbar_warmup_mean": float(c_bar.mean().item()),
        }
        return L_fuzzy, weights, stats

    def _inter_loss(self,
                    view1_feature: torch.Tensor,
                    view2_feature: torch.Tensor,
                    labels: torch.Tensor):
        margin = 0.2
        shift = 1.0
        tau = 1.0

        u = view1_feature.tanh()
        v = view2_feature.tanh()

        B = u.size(0)
        label1 = labels.float()
        label2 = labels.float()
        T = (label1 @ label2.t() > 0).float()
        T.fill_diagonal_(0.0)

        S = u.mm(v.t())  # [B,B]

        d = S.diag().view(B, 1)
        d1 = d.expand_as(S)
        d2 = d.t().expand_as(S)

        mask_te = (S >= (d1 - margin)).float().detach()
        cost_te = S * mask_te + (1.0 - mask_te) * (S - shift)
        cost_te_max = cost_te.clone()

        eye = torch.eye(B, device=S.device, dtype=S.dtype)
        diag_te = torch.diag(cost_te_max).clamp(min=0.0)
        cost_te_max = cost_te_max * (1 - eye) + torch.diag_embed(diag_te)

        mask_im = (S >= (d2 - margin)).float().detach()
        cost_im = S * mask_im + (1.0 - mask_im) * (S - shift)
        cost_im_max = cost_im.clone()
        diag_im = torch.diag(cost_im_max).clamp(min=0.0)
        cost_im_max = cost_im_max * (1 - eye) + torch.diag_embed(diag_im)

        neg_mask = (1.0 - T)
        exp_te = (cost_te_max / tau * neg_mask).exp()
        exp_im = (cost_im_max / tau * neg_mask).exp()
        sum_te = exp_te.sum(dim=1).clamp_min(1e-6)
        sum_im = exp_im.sum(dim=1).clamp_min(1e-6)

        term_te = -cost_te.diag() + tau * sum_te.log() + margin
        term_im = -cost_im.diag() + tau * sum_im.log() + margin

        loss_r = term_te + term_im
        return loss_r.mean()

    def _build_far_weights(self,
                           view1_predict: torch.Tensor,
                           view2_predict: torch.Tensor,
                           labels: torch.Tensor):
        device = labels.device
        B = labels.size(0)
        eps = 1e-8

        p1 = self._normalize_rows(view1_predict)
        p2 = self._normalize_rows(view2_predict)

        gamma_f = 1.0
        if gamma_f > 1.0:
            p1 = self._sharpen_probs(p1, gamma=gamma_f)
            p2 = self._sharpen_probs(p2, gamma=gamma_f)

        a1 = get_train_category_credibility(p1, labels, comp_topk=self.comp_topk)
        a2 = get_train_category_credibility(p2, labels, comp_topk=self.comp_topk)
        a_bar = 0.5 * (a1 + a2)  # [B, K]

        t_idx = torch.argmax(labels, dim=1)  # [B]
        ar = torch.arange(B, device=device)

        pos_score = a_bar[ar, t_idx]  # [B]
        a_bar_mask = a_bar.clone()
        a_bar_mask[ar, t_idx] = -1e9
        neg_score = a_bar_mask.max(dim=1).values  # [B]

        gap = pos_score - neg_score  # [B]

        gap = (gap - gap.mean()) / (gap.std(unbiased=False) + eps)

        delta = 0.0
        temp = 0.7

        weights = torch.sigmoid((gap - delta) / temp)
        return weights.clamp(0.0, 1.0).detach(), a_bar.detach()



    def _far_margin_loss(self,
                         view1_feature: torch.Tensor,
                         view2_feature: torch.Tensor,
                         labels: torch.Tensor,
                         sample_weights: torch.Tensor):
        device = view1_feature.device
        B = view1_feature.size(0)
        if B <= 1:
            return torch.zeros((), device=device)

        gamma = 0.20

        z1 = F.normalize(view1_feature.tanh(), dim=1)
        z2 = F.normalize(view2_feature.tanh(), dim=1)

        S = (z1 @ z2.t()).clamp(-1.0, 1.0)  # [B,B]

        t_idx = torch.argmax(labels, dim=1)  # [B]
        zeta = torch.where(
            t_idx.view(B, 1) == t_idx.view(1, B),
            torch.ones(B, B, device=device, dtype=S.dtype),
            -torch.ones(B, B, device=device, dtype=S.dtype)
        )

        w = sample_weights.to(device=device, dtype=S.dtype).view(B)
        pair_w = torch.min(w.view(B, 1), w.view(1, B))  # [B,B]

        margin_term = torch.relu(gamma - zeta * S)

        eye = torch.eye(B, device=device, dtype=torch.bool)
        valid_mask = ~eye

        numer = (pair_w[valid_mask] * margin_term[valid_mask]).sum()
        denom = pair_w[valid_mask].sum().clamp_min(1e-8)

        return numer / denom

    def forward(self,
                view1_feature, view2_feature,
                view1_predict, view2_predict,
                labels, epoch, opt):
        alpha = float(opt.alpha)
        tau = float(opt.tau)
        q = float(opt.q)

        cur_epoch = int(epoch) + 1
        tp = int(opt.tp)

        cbar_warmup_min = cbar_warmup_max = cbar_warmup_mean = -1.0
        cbar_finetune_min = cbar_finetune_max = cbar_finetune_mean = -1.0

        L_FAR = torch.zeros((), device=view1_feature.device)
        L_fuzzy = torch.zeros((), device=view1_feature.device)
        L_inter = torch.zeros((), device=view1_feature.device)

        if cur_epoch <= tp:
            stage_tag = "Warm-up"
            L_fuzzy, weights, st = self._warmup_fuzzy_loss(
                view1_predict, view2_predict, labels
            )
            cbar_warmup_min = st["cbar_warmup_min"]
            cbar_warmup_max = st["cbar_warmup_max"]
            cbar_warmup_mean = st["cbar_warmup_mean"]

        else:
            stage_tag = "Fine-tuning"

            L_inter = self._inter_loss(
                view1_feature, view2_feature, labels
            )

            sample_weights, a_bar_stat = self._build_far_weights(
                view1_predict=view1_predict,
                view2_predict=view2_predict,
                labels=labels
            )

            cbar_finetune_min = float(a_bar_stat.min().item())
            cbar_finetune_max = float(a_bar_stat.max().item())
            cbar_finetune_mean = float(a_bar_stat.mean().item())

            L_FAR = self._far_margin_loss(
                view1_feature=view1_feature,
                view2_feature=view2_feature,
                labels=labels,
                sample_weights=sample_weights
            )
            weights = sample_weights

        L_hash = ((view1_feature.abs() - 1.0).pow(2).mean() +
                  (view2_feature.abs() - 1.0).pow(2).mean())

        L_intra = intra_loss(
            [view1_feature, view2_feature], tau=tau, q=q
        )

        if cur_epoch <= tp:
            loss = L_fuzzy + alpha * L_intra
        else:
            loss = L_FAR + alpha * L_intra + L_inter

        info = {
            "stage": stage_tag,
            "L_fuzzy": float(L_fuzzy.item()),
            "L_intra": float(L_intra.item()),
            "L_hash": float(L_hash.item()),
            "L_inter": float(L_inter.item()) if cur_epoch > tp else 0.0,
            "L_FAR": float(L_FAR.item()) if cur_epoch > tp else 0.0,
            "cbar_warmup_min": float(cbar_warmup_min),
            "cbar_warmup_max": float(cbar_warmup_max),
            "cbar_warmup_mean": float(cbar_warmup_mean),
            "cbar_finetune_min": float(cbar_finetune_min),
            "cbar_finetune_max": float(cbar_finetune_max),
            "cbar_finetune_mean": float(cbar_finetune_mean),
            "weights": weights if weights is not None else torch.ones(labels.size(0), device=labels.device),
        }
        return loss, info
