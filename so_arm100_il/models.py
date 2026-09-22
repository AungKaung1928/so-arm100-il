"""Three policies, one loss.

    StateMLP      privileged 25-d state is NOT used here; the 15-d proprio
                  vector alone cannot locate the cube, so this model exists
                  as the floor: what does a policy learn from proprioception
                  and nothing else. Its success rate is the number the image
                  models have to beat.
    ImagePolicy   a small CNN whose last layer is a spatial soft-argmax over
                  K keypoint channels. The layer *is* a coordinate: position
                  is handed to the network instead of learned, which is the
                  visuomotor front end of Levine et al. and the head that won
                  in an earlier pose-regression project on this same laptop.
                  Keypoints concatenated with proprio feed an MLP.
    ACTLite       Action Chunking Transformer without the CVAE. Tokens are
                  the K keypoints (each (u, v) embedded plus a learned channel
                  embedding) and the proprio vector; a 2-layer encoder reads
                  them, a 2-layer decoder turns k learned queries into a chunk
                  of k future actions. At run time chunks overlap and are
                  temporally ensembled (`TemporalEnsembler`).

Loss is L1 on normalised actions for all three. MSE fits the conditional
mean; when demonstrations are multimodal (two valid grasp yaws, two ways
round a cube) the mean is a third, invalid action. L1 fits the conditional
median, which is one of the modes, and it is what ACT trains with. Padded
chunk entries are masked out.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import STATE_DIM, ACT_DIM


def masked_l1(pred, target, mask):
    """pred/target (B, k, 6), mask (B, k) True where padded."""
    keep = (~mask).float().unsqueeze(-1)
    return (torch.abs(pred - target) * keep).sum() / keep.sum().clamp(min=1.0) / pred.shape[-1]


def conv_block(cin, cout, stride=2):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SpatialSoftArgmax(nn.Module):
    """(B, K, H, W) -> (B, K, 2) expected (u, v) per channel in [-1, 1]."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, x):
        b, k, h, w = x.shape
        p = F.softmax(x.reshape(b, k, h * w) / self.log_t.exp(), dim=-1).reshape(b, k, h, w)
        u = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        v = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        return torch.stack([(p * u).sum((2, 3)), (p * v).sum((2, 3))], dim=-1)


class KeypointEncoder(nn.Module):
    """uint8 (B, 3, H, W) -> (B, K, 2) keypoints. Stride 8 trunk."""

    def __init__(self, keypoints=16):
        super().__init__()
        self.trunk = nn.Sequential(conv_block(3, 16), conv_block(16, 32), conv_block(32, 64))
        self.kp = nn.Conv2d(64, keypoints, 1)
        self.ssam = SpatialSoftArgmax()
        self.keypoints = keypoints

    def forward(self, img):
        x = img.float() / 255.0
        x = (x - 0.45) / 0.25
        return self.ssam(self.kp(self.trunk(x)))


def mlp(cin, hidden, cout):
    return nn.Sequential(nn.Linear(cin, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
                         nn.Linear(hidden, cout))


class StateMLP(nn.Module):
    kind = "state"
    chunk = 1
    needs_image = False

    def __init__(self, hidden=256):
        super().__init__()
        self.net = mlp(STATE_DIM, hidden, ACT_DIM)

    def forward(self, state, image=None):
        return self.net(state).unsqueeze(1)          # (B, 1, 6)


class ImagePolicy(nn.Module):
    kind = "image"
    chunk = 1
    needs_image = True

    def __init__(self, keypoints=16, hidden=256):
        super().__init__()
        self.enc = KeypointEncoder(keypoints)
        self.head = mlp(2 * keypoints + STATE_DIM, hidden, ACT_DIM)

    def forward(self, state, image):
        kp = self.enc(image).flatten(1)
        return self.head(torch.cat([kp, state], dim=1)).unsqueeze(1)


class ACTLite(nn.Module):
    kind = "act"
    needs_image = True

    def __init__(self, chunk=20, keypoints=16, d_model=128, nhead=4, layers=2, ff=256):
        super().__init__()
        self.chunk = int(chunk)
        self.enc = KeypointEncoder(keypoints)
        self.kp_proj = nn.Linear(2, d_model)
        self.kp_embed = nn.Parameter(torch.randn(keypoints, d_model) * 0.02)
        self.state_proj = nn.Linear(STATE_DIM, d_model)
        self.state_embed = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.queries = nn.Parameter(torch.randn(self.chunk, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout=0.0, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)
        self.out = nn.Linear(d_model, ACT_DIM)

    def forward(self, state, image):
        kp = self.enc(image)                                            # (B, K, 2)
        tokens = torch.cat([self.kp_proj(kp) + self.kp_embed,
                            (self.state_proj(state) + self.state_embed).unsqueeze(1)], dim=1)
        mem = self.encoder(tokens)
        q = self.queries.unsqueeze(0).expand(state.shape[0], -1, -1)
        return self.out(self.decoder(q, mem))                           # (B, k, 6)


def build(kind, **kw):
    return {"state": StateMLP, "image": ImagePolicy, "act": ACTLite}[kind](**kw)


class TemporalEnsembler:
    """Combine overlapping action chunks at run time.

    At step t the policy has predicted the action for t from every chunk
    issued in the last k steps. They are averaged with weights exp(-m * age)
    where age = 0 is the newest chunk; m = 0 is a plain mean, larger m trusts
    the newest prediction more. The weights are normalised, so with a single
    prediction in the buffer the output is exactly that prediction.
    """

    def __init__(self, chunk, m=0.1):
        self.chunk, self.m = int(chunk), float(m)
        self.reset()

    def reset(self):
        self._preds = []        # list of (issued_t, chunk array (k, 6))
        self.t = 0

    def weights(self, n):
        w = torch.exp(-self.m * torch.arange(n, dtype=torch.float32))
        return w / w.sum()

    def step(self, chunk_pred):
        """chunk_pred (k, 6) issued now; returns the action for the current step."""
        self._preds.append((self.t, chunk_pred))
        self._preds = [(t0, c) for t0, c in self._preds if self.t - t0 < self.chunk]
        cands = [c[self.t - t0] for t0, c in reversed(self._preds)]      # newest first
        stack = torch.as_tensor(torch.stack([torch.as_tensor(c) for c in cands]))
        w = self.weights(len(cands)).unsqueeze(1)
        self.t += 1
        return (w * stack).sum(0)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def orthogonal_init(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, math.sqrt(2.0))
            nn.init.zeros_(m.bias)
