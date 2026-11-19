"""Stage 2: train the template-warping mask head(s) on labeled frames, on top of a pretrained world model.

Each step trains the head on two inputs from the same clip:
  tracking  : true tokens of the current frame t        -> mask at t
  forecast  : world-model tokens predicted for t + k    -> mask at t + k
M heads (different inits) form the ensemble used for uncertainty.

Exploratory DINOv3 baseline in the paper (predictor and heads trained end to end on the 40 labeled scans):
    python p1_trackrad/train_head.py --name pilot_e2e --wm none --M 3 --steps 4000 --ft_pred 1e-4
"""
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader

from common import RUNS, JsonLog, metas_for, seed_all
from data import ClipDataset
from models import FrozenEncoder, MaskHead, Predictor, seg_loss, smooth_loss

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--wm", required=True, help="stage-1 run name, or 'none' for a fresh predictor (needs --ft_pred)")
ap.add_argument("--M", type=int, default=5)
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--ft_pred", type=float, default=0.0, help="predictor LR (0 = frozen)")
ap.add_argument("--w_smooth", type=float, default=1.0)
ap.add_argument("--val_every", type=int, default=500)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

seed_all(args.seed)
run = RUNS / args.name
run.mkdir(parents=True, exist_ok=True)
(run / "args.json").write_text(json.dumps(vars(args), indent=1))
log = JsonLog(run / "log.jsonl")

if args.wm == "none":
    assert args.ft_pred > 0, "a fresh predictor must be trained (--ft_pred)"
    wa = dict(K=8, ctx_window_s=5.0, max_ahead_s=1.5, d=384, depth=6)
else:
    ck = torch.load(RUNS / args.wm / "pred.pt", map_location="cpu")
    wa = ck["args"]
enc = FrozenEncoder().cuda()
pred = Predictor(d=wa["d"], depth=wa["depth"], heads=wa["d"] // 64).cuda()
if args.wm != "none":
    pred.load_state_dict(ck["pred"])
pred.requires_grad_(args.ft_pred > 0).train(args.ft_pred > 0)
heads = torch.nn.ModuleList([MaskHead() for _ in range(args.M)]).cuda()

groups = [{"params": heads.parameters(), "lr": args.lr}]
if args.ft_pred > 0:
    groups.append({"params": pred.parameters(), "lr": args.ft_pred})
opt = torch.optim.AdamW(groups, weight_decay=0.05)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in groups], total_steps=args.steps, pct_start=0.05)

mk = lambda split, n, seed: ClipDataset(metas_for(split), K=wa["K"], ctx_window_s=wa["ctx_window_s"],
                                        max_ahead_s=wa["max_ahead_s"], samples_per_epoch=n, seed=seed, labels=True)
loader = DataLoader(mk("fit", args.steps * args.bs, args.seed), args.bs, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
val_batches = list(DataLoader(mk("val", 512, 123), args.bs, num_workers=4))


def forward(b):
    B, K = b["ctx"].shape[:2]
    z = enc(torch.cat([b["ctx"].flatten(0, 1), b["ref_img"]]).cuda(non_blocking=True))
    z_ctx, z_ref = z[:B * K].view(B, K, *z.shape[1:]), z[B * K:]
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.set_grad_enabled(args.ft_pred > 0 and pred.training):
        z_hat = pred(z_ctx, b["tau_ctx"].cuda(), b["tau_tgt"].cuda())[:, 0].float()
    ref_mask = b["ref_mask"].cuda()
    out = []
    for h in heads:
        lt, ft = h(z_ctx[:, -1], z_ref, ref_mask)
        lf, ff = h(z_hat, z_ref, ref_mask)
        out.append((lt, lf, ft, ff))
    return out


def dice(logits, t):
    p = (logits > 0).float()
    t = t.float()
    return ((2 * (p * t).flatten(1).sum(1) + 1e-6) / (p.flatten(1).sum(1) + t.flatten(1).sum(1) + 1e-6)).mean().item()


@torch.no_grad()
def validate():
    heads.eval(), pred.eval()
    s = {"dice_track": 0.0, "dice_hold": 0.0, "dice_fcst": 0.0, "dice_fcst_ens": 0.0}
    for b in val_batches:
        out = forward(b)
        cur, tgt = b["cur_mask"].cuda(), b["tgt_mask"].cuda()
        s["dice_track"] += sum(dice(o[0], cur) for o in out) / len(out)
        s["dice_hold"] += sum(dice(o[0], tgt) for o in out) / len(out)
        s["dice_fcst"] += sum(dice(o[1], tgt) for o in out) / len(out)
        s["dice_fcst_ens"] += dice(torch.stack([o[1].sigmoid() for o in out]).mean(0).logit(1e-6), tgt)
    heads.train(), pred.train(args.ft_pred > 0)
    return {k: v / len(val_batches) for k, v in s.items()}


t0, step = time.time(), 0
for b in loader:
    out = forward(b)
    cur, tgt = b["cur_mask"].cuda(), b["tgt_mask"].cuda()
    loss = sum(seg_loss(lt, cur) + seg_loss(lf, tgt) + args.w_smooth * (smooth_loss(ft) + smooth_loss(ff))
               for lt, lf, ft, ff in out) / len(out)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(), sched.step()
    step += 1
    if step % 100 == 0:
        log(step=step, loss=loss.item(), it_s=100 / (time.time() - t0))
        t0 = time.time()
    if step % args.val_every == 0 or step == args.steps:
        log(step=step, **validate())
        torch.save({"heads": heads.state_dict(), "pred": pred.state_dict(), "wm": args.wm, "wm_args": wa,
                    "args": vars(args), "step": step}, run / "head.pt")
    if step >= args.steps:
        break
print("done", flush=True)
