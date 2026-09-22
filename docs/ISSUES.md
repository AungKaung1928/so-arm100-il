# Issues to open on GitHub

One block per issue: title, then body. The known gaps at the first release.

---

**ACT with the CVAE latent**

`ACTLite` drops the CVAE half of ACT: no style variable, no KL term. On
scripted demonstrations the action given the observation is close to
deterministic, so the latent has little to encode; on human or human-video
demonstrations that stops being true. Add `--cvae` with a KL weight and
compare on the same dataset before claiming either way.

---

**Wrist camera for the image policies**

Only the fixed `front` camera is used. A wrist view makes grasp alignment a
local problem; the bench has no wrist camera yet (tracked there). When it
lands, add `--camera` to `train_bc.py` and repeat the scaling curve.

---

**Validate the human-video path on a real recording**

`scripts/hand_to_arm.py` has only been exercised on synthetic landmarks. Run
it on an overhead phone video of a pinch-and-lift, report how many frames
detect a hand, whether the replay succeeds, and what the mapping constants
(`PINCH_CLOSED`, `PINCH_OPEN`, the fixed scale) had to become.

---

**DAgger with a decaying beta needs a stopping rule**

Iterations are a fixed count. The natural rule -- stop when learner success
on the small check has not improved for two iterations -- is not
implemented, so `--iters` is chosen by hand.

---

**Image policies evaluate slowly on software GL**

100 episodes x 5 seeds x ~100 frames at 20 ms per render is about 17
minutes per physics cell for an image policy, before the policy's own
forward pass. The held-out table is seven cells. Either cache renders per
seed or accept the wall clock; measure and write it down.

---

**Scaling curve on tasks other than lift**

`scaling.py` defaults to `lift:red`. Push and pick_place have different
demonstration lengths and failure modes; the curve's shape may not transfer.
