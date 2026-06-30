"""
Gradient-flow verification test for the tortuosity loss.

Run from inside 5_TAU/ (no GPU required):
  conda run -n ganph python test_tau_loss.py

What this checks:
  1. TauNet forward pass works on random input.
  2. Gradient flows from tau_loss all the way back to a mock generator
     parameter — this is THE invariant that BUG 1 broke in _loss_ssa.
  3. tau_net parameters have zero gradient (correctly frozen).
  4. _loss_tortuosity skips phases with None targets gracefully.
  5. Existing training (no tau_net) is unaffected.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))           # 5_TAU/
sys.path.insert(0, str(Path(__file__).parent.parent / "1_GAN"))  # 1_GAN/

from tau_net import TauNet


# ── Test 1: TauNet forward pass ───────────────────────────────────────────────
def test_forward():
    net = TauNet(ndf=8)   # tiny for speed
    x   = torch.randn(2, 1, 64, 64, 64)
    out = net(x)
    assert out.shape == (2,), f"Expected shape (2,), got {out.shape}"
    print("PASS  test_forward: output shape correct")


# ── Test 2: Gradient flows through tau_net to the input ──────────────────────
def test_gradient_flows_to_input():
    """
    Simulate exactly what _loss_tortuosity does:
      phase_prob = g_data[:, ch:ch+1, :, :, :]   # view of g_data
      tau_pred   = tau_net(phase_prob)             # forward, NO no_grad
      loss       = (tau_pred - target)**2
      loss.backward()                              # grad reaches g_data
    """
    net = TauNet(ndf=8)
    for p in net.parameters():
        p.requires_grad_(False)   # frozen, as in Trainer.__init__
    net.eval()

    g_data = torch.randn(2, 3, 64, 64, 64, requires_grad=True)

    # Simulate the Ni channel loss
    phase_prob = g_data[:, 0:1, :, :, :]
    tau_pred   = net(phase_prob)
    target     = torch.tensor(2.5)
    loss       = torch.mean((tau_pred - target) ** 2)
    loss.backward()

    assert g_data.grad is not None, "g_data.grad is None — gradient severed!"
    assert g_data.grad.abs().sum() > 0, "g_data.grad is all zeros — no signal"
    print(f"PASS  test_gradient_flows_to_input: "
          f"|grad|={g_data.grad.abs().mean():.4e}")


# ── Test 3: tau_net parameters remain frozen ─────────────────────────────────
def test_tau_net_params_frozen():
    net = TauNet(ndf=8)
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()

    g_data = torch.randn(2, 3, 64, 64, 64, requires_grad=True)
    phase_prob = g_data[:, 0:1, :, :, :]
    loss = torch.mean((net(phase_prob) - 2.5) ** 2)
    loss.backward()

    for name, p in net.named_parameters():
        assert p.grad is None, f"tau_net param {name} accumulated grad — NOT frozen!"
    print("PASS  test_tau_net_params_frozen: all τ-net params have grad=None")


# ── Test 4: _loss_tortuosity via real Trainer ─────────────────────────────────
def test_loss_tortuosity_in_trainer():
    """
    Import Trainer, construct a minimal instance, call _loss_tortuosity.
    Verifies the method exists and runs without error.
    """
    import training   # from 1_GAN/

    # Minimal stubs — we only call _loss_tortuosity, not train()
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Linear(1, 1)   # gives Adam a non-empty param list
        def forward(self, *a): return torch.zeros(1)

    net = TauNet(ndf=8)
    targets = {"Ni": 2.5, "YSZ": 3.0, "Pore": 1.8}

    trainer = training.Trainer(
        model_g    = FakeModel(),
        optim_g    = torch.optim.Adam(FakeModel().parameters()),
        model_c    = FakeModel(),
        optim_c    = torch.optim.Adam(FakeModel().parameters()),
        model_e    = FakeModel(),
        epochs     = 1,
        device     = torch.device("cpu"),
        dataloader = [],
        in_header  = ".",
        tau_net    = net,
        tau_targets= targets,
    )

    g_data = torch.randn(2, 3, 64, 64, 64, requires_grad=True)
    loss   = trainer._loss_tortuosity(g_data)

    assert loss.requires_grad, "tau_loss has no grad — backward will be silent"
    loss.backward()
    assert g_data.grad is not None and g_data.grad.abs().sum() > 0
    print(f"PASS  test_loss_tortuosity_in_trainer: loss={loss.item():.4f}, "
          f"|grad|={g_data.grad.abs().mean():.4e}")


# ── Test 5: None targets skipped gracefully ───────────────────────────────────
def test_none_targets_skipped():
    import training

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Linear(1, 1)   # gives Adam a non-empty param list
        def forward(self, *a): return torch.zeros(1)

    net = TauNet(ndf=8)
    # Only Pore has a target; Ni and YSZ are None → should be skipped
    targets = {"Pore": 1.8}

    trainer = training.Trainer(
        model_g=FakeModel(), optim_g=torch.optim.Adam(FakeModel().parameters()),
        model_c=FakeModel(), optim_c=torch.optim.Adam(FakeModel().parameters()),
        model_e=FakeModel(), epochs=1, device=torch.device("cpu"),
        dataloader=[], in_header=".", tau_net=net, tau_targets=targets,
    )
    g_data = torch.randn(2, 3, 64, 64, 64, requires_grad=True)
    loss   = trainer._loss_tortuosity(g_data)
    loss.backward()
    assert g_data.grad is not None
    print(f"PASS  test_none_targets_skipped: partial targets work, loss={loss.item():.4f}")


# ── Test 6: tau_net=None leaves G_loss unchanged ─────────────────────────────
def test_no_tau_net_unchanged():
    """
    When tau_net is None, _loss_tortuosity should never be called in train().
    Verify the losses dict has an empty tau_loss list after __init__.
    """
    import training

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._p = nn.Linear(1, 1)   # gives Adam a non-empty param list
        def forward(self, *a): return torch.zeros(1)

    trainer = training.Trainer(
        model_g=FakeModel(), optim_g=torch.optim.Adam(FakeModel().parameters()),
        model_c=FakeModel(), optim_c=torch.optim.Adam(FakeModel().parameters()),
        model_e=FakeModel(), epochs=1, device=torch.device("cpu"),
        dataloader=[], in_header=".",
        # no tau_net, no tau_targets
    )
    assert trainer.tau_net is None
    assert trainer.losses["tau_loss"] == []
    print("PASS  test_no_tau_net_unchanged: tau_net=None leaves losses dict clean")


# ── Runner ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running τ-loss gradient flow tests ...\n")
    test_forward()
    test_gradient_flows_to_input()
    test_tau_net_params_frozen()
    test_loss_tortuosity_in_trainer()
    test_none_targets_skipped()
    test_no_tau_net_unchanged()
    print("\nAll tests passed.")
