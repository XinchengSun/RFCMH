import torch
import torch.nn as nn
import torch.nn.functional as F


class ImgNN(nn.Module):
    def __init__(self, input_dim=4096, output_dim=128, tanh=True):
        super().__init__()
        hidden_dim = 4096
        self.tanh = tanh

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

        bound = 1.0 / (float(input_dim) ** 0.5)
        nn.init.uniform_(self.fc3.weight, -bound, bound)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if self.tanh:
            out = out.tanh()
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-12)


class TextNN(nn.Module):
    def __init__(self, input_dim=1000, output_dim=128, tanh=True):
        super().__init__()
        hidden_dim = 4096
        self.tanh = tanh

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)

        bound = 1.0 / (float(input_dim) ** 0.5)
        nn.init.uniform_(self.fc3.weight, -bound, bound)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if self.tanh:
            out = out.tanh()
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-12)


class IDCM_NN(nn.Module):
    def __init__(
        self,
        img_input_dim=4096,
        text_input_dim=1000,
        output_dim=128,
        num_class=100,
        tanh=True,
        member_tau=0.5,
        member_gamma=2.0,
    ):
        super().__init__()
        self.member_tau = float(member_tau)
        self.member_gamma = float(member_gamma)

        self.img_net = ImgNN(img_input_dim, output_dim=output_dim, tanh=tanh)
        self.text_net = TextNN(text_input_dim, output_dim=output_dim, tanh=tanh)

        if output_dim >= num_class:
            w_full = torch.empty(output_dim, output_dim)
            w_full = torch.nn.init.orthogonal_(w_full, gain=1)
            w = F.normalize(w_full[:, :num_class], p=2, dim=0)
        else:
            w = self._init_w_spread(D=output_dim, K=num_class, refine_steps=50, eta=0.05)

        self.register_buffer("W", w)
        self._verify_w()

    def _init_w_spread(self, D, K, refine_steps=50, eta=0.05, device=None, dtype=None):
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        w = torch.randn(D, K, device=device, dtype=dtype)
        w = F.normalize(w, p=2, dim=0)
        for _ in range(refine_steps):
            gram = w.t() @ w
            gram.fill_diagonal_(0.0)
            w = F.normalize(w - eta * (w @ gram), p=2, dim=0)
        return w

    def _W_col_norm(self):
        return F.normalize(self.W, p=2, dim=0)

    def _verify_w(self):
        with torch.no_grad():
            w = self._W_col_norm()
            gram = w.t() @ w
            eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            orth_error = torch.norm(gram - eye, p="fro").item()
            col_norm = torch.norm(w, dim=0)
            print(
                f"[Model Init] W shape: {w.shape}, "
                f"orth_error={orth_error:.6f}, "
                f"col-norm min={col_norm.min():.4f}, max={col_norm.max():.4f}"
            )

    def membership(self, h):
        proj = h @ self._W_col_norm()
        membership = torch.sigmoid(proj / max(self.member_tau, 1e-6))
        if self.member_gamma != 1.0:
            membership = membership.pow(self.member_gamma)
        return membership

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)
        return view1_feature, view2_feature
