"""Tests for the Kronecker-factored (nearest Kronecker product) curvature estimate."""

import pytest
import torch
from torch import nn

from nanoquant.core import importance as imp


def _vec(G: torch.Tensor) -> torch.Tensor:
    """Column-major vectorisation, so that vec(delta x^T) = x kron delta."""
    return G.mT.reshape(-1)


def _empirical_fisher(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """F = sum_t vec(G_t) vec(G_t)^T with G_t = delta_t x_t^T."""
    vecs = torch.stack([_vec(torch.outer(d, xx)) for xx, d in zip(x, delta)])
    return vecs.mT @ vecs


def _rearrange(F: torch.Tensor, n_in: int, n_out: int) -> torch.Tensor:
    """Van Loan rearrangement: kron(R, L) -> vec(R) vec(L)^T, shape (in^2, out^2)."""
    return F.reshape(n_in, n_out, n_in, n_out).permute(0, 2, 1, 3).reshape(n_in * n_in, n_out * n_out)


def _scaled_residual(F: torch.Tensor, L: torch.Tensor, R: torch.Tensor) -> float:
    """min_c ||F - c kron(R, L)||_F^2."""
    K = torch.kron(R, L)
    return (F.square().sum() - (F * K).sum().square() / K.square().sum()).item()


def _fix_sign(M: torch.Tensor) -> torch.Tensor:
    return M if torch.trace(M) >= 0 else -M


def test_nkp_update_from_identity_equals_shampoo_factors():
    torch.manual_seed(0)
    T, n_in, n_out = 20, 4, 3
    x = torch.randn(T, n_in)
    delta = torch.randn(T, n_out)

    L, R = imp.nkp_update(x, delta, L_prev=None, R_prev=None)

    G = torch.einsum("to,ti->toi", delta, x)  # (T, out, in)
    L_ref = torch.einsum("toi,tpi->op", G, G)  # sum_t G_t G_t^T
    R_ref = torch.einsum("toi,toj->ij", G, G)  # sum_t G_t^T G_t
    assert torch.allclose(L, L_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(R, R_ref, atol=1e-5, rtol=1e-5)


def test_nkp_update_uses_previous_factors_as_weights():
    torch.manual_seed(1)
    T, n_in, n_out = 10, 4, 3
    x = torch.randn(T, n_in)
    delta = torch.randn(T, n_out)
    A = torch.randn(n_in, n_in)
    R_prev = A @ A.mT
    B = torch.randn(n_out, n_out)
    L_prev = B @ B.mT

    L, R = imp.nkp_update(x, delta, L_prev=L_prev, R_prev=R_prev)

    w_L = torch.einsum("ti,ij,tj->t", x, R_prev, x)
    w_R = torch.einsum("to,op,tp->t", delta, L_prev, delta)
    L_ref = torch.einsum("t,to,tp->op", w_L, delta, delta)
    R_ref = torch.einsum("t,ti,tj->ij", w_R, x, x)
    assert torch.allclose(L, L_ref, atol=1e-4, rtol=1e-5)
    assert torch.allclose(R, R_ref, atol=1e-4, rtol=1e-5)


def test_nkp_fit_converges_to_van_loan_solution():
    torch.manual_seed(2)
    T, n_in, n_out = 40, 3, 4
    # correlated inputs / gradients so the Kronecker structure is non-trivial
    x = torch.randn(T, n_in) @ torch.tensor([[1.0, 0.8, 0.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    delta = torch.randn(T, n_out) @ torch.triu(torch.ones(n_out, n_out))
    F = _empirical_fisher(x, delta)

    residuals = []
    for iters in (1, 3, 6, 12):
        L, R = imp.nkp_fit(x, delta, num_iters=iters)
        assert abs(L.norm().item() - 1.0) < 1e-5 and abs(R.norm().item() - 1.0) < 1e-5
        residuals.append(_scaled_residual(F, L, R))
    # the Jacobi (both-factors-per-pass) iteration is not guaranteed monotone step to step,
    # but it must improve on the Shampoo-like first pass and converge
    assert residuals[-1] < residuals[0]
    assert residuals[-1] <= residuals[-2] + 1e-6 * F.square().sum().item()

    # fixed point == top singular pair of the rearranged Fisher (Van Loan & Pitsianis)
    U, _S, Vh = torch.linalg.svd(_rearrange(F, n_in, n_out), full_matrices=False)
    R_star = _fix_sign(U[:, 0].reshape(n_in, n_in))
    L_star = _fix_sign(Vh[0].reshape(n_out, n_out))
    L, R = imp.nkp_fit(x, delta, num_iters=40)
    assert torch.allclose(_fix_sign(L), L_star, atol=2e-3)
    assert torch.allclose(_fix_sign(R), R_star, atol=2e-3)


def test_dense_shrinkage_generalises_vector_shrinkage():
    torch.manual_seed(3)
    n_in, n_out, s = 5, 4, 0.4
    A = torch.randn(n_in, n_in)
    B = torch.randn(n_out, n_out)
    i_cov = A @ A.mT
    o_cov = B @ B.mT
    raw = {
        "i_norm": {"l": i_cov.diagonal().clone()},
        "o_norm": {"l": o_cov.diagonal().clone()},
        "i_cov": {"l": i_cov.clone()},
        "o_cov": {"l": o_cov.clone()},
        "stats_device": "cpu",
    }
    shrunk = imp.get_shrunk_stats(raw, shrinkage=s)

    # raw stats untouched
    assert torch.equal(raw["i_cov"]["l"], i_cov)

    for key_vec, key_cov, cov in (("i_norm", "i_cov", i_cov), ("o_norm", "o_cov", o_cov)):
        d = cov.diagonal()
        expected_diag = (1 - s) * d + s * d.mean()
        got = shrunk[key_cov]["l"]
        assert torch.allclose(got.diagonal(), expected_diag, atol=1e-6)
        off = ~torch.eye(cov.shape[0], dtype=torch.bool)
        assert torch.allclose(got[off], (1 - s) * cov[off], atol=1e-6)
        # preconditioning diagonal is the shrunk dense diagonal
        assert torch.allclose(shrunk[key_vec]["l"], got.diagonal(), atol=1e-6)


def test_dense_shrinkage_noop_without_dense_keys():
    raw = {"i_norm": {"l": torch.ones(3)}, "o_norm": {"l": torch.ones(2)}, "stats_device": "cpu"}
    shrunk = imp.get_shrunk_stats(raw, shrinkage=0.4)
    assert "i_cov" not in shrunk and "o_cov" not in shrunk


class _TinyMLP(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(6, 5, bias=False), nn.Tanh(), nn.Linear(5, 3, bias=False))


def _fake_loop(dataloader, model, dev, model_offload, use_truefisher):
    for batch in dataloader:
        inp = batch.clone().requires_grad_(True)
        loss = model(inp).square().mean()
        loss.backward()
        model.zero_grad(set_to_none=True)


@pytest.mark.parametrize("strategy", ["online", "dbf"])
def test_collect_stats_kron_on_tiny_mlp(monkeypatch, strategy):
    torch.manual_seed(4)
    model = _TinyMLP()
    dataloader = [torch.randn(2, 7, 6) for _ in range(3)]
    calls = {"n": 0}

    def counting_loop(*args, **kwargs):
        calls["n"] += 1
        _fake_loop(*args, **kwargs)

    monkeypatch.setattr(imp, "_run_calibration_loop", counting_loop)

    raw = imp.collect_stats(model, dataloader, "cpu", strategy=strategy, curvature="kron", nkp_iters=3)

    assert calls["n"] == 3
    assert raw["stats_device"] == "cpu"
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        L = raw["o_cov"][name]
        R = raw["i_cov"][name]
        assert L.shape == (m.out_features, m.out_features)
        assert R.shape == (m.in_features, m.in_features)
        assert L.dtype == torch.float32 and R.dtype == torch.float32
        assert torch.allclose(L, L.mT, atol=1e-6) and torch.allclose(R, R.mT, atol=1e-6)
        assert torch.linalg.eigvalsh(L).min() > -1e-6 and torch.linalg.eigvalsh(R).min() > -1e-6
        assert L.norm() > 0 and R.norm() > 0
        assert torch.allclose(raw["o_norm"][name], L.diagonal())
        assert torch.allclose(raw["i_norm"][name], R.diagonal())


def test_collect_stats_kron_matches_offline_fit(monkeypatch):
    """Streaming multi-pass estimate equals the in-memory ALS fit on the same tokens (no clipping)."""
    torch.manual_seed(5)
    model = _TinyMLP()
    dataloader = [torch.randn(1, 9, 6) for _ in range(2)]
    monkeypatch.setattr(imp, "_run_calibration_loop", _fake_loop)

    # gather the exact tokens seen by the first layer
    xs, ds = [], []
    lin = model[0]
    h1 = lin.register_forward_hook(lambda m, i, o: xs.append(i[0].detach().flatten(0, -2).float()))
    h2 = lin.register_full_backward_hook(lambda m, gi, go: ds.append(go[0].detach().flatten(0, -2).float()))
    _fake_loop(dataloader, model, "cpu", False, False)
    h1.remove()
    h2.remove()
    x = torch.cat(xs)
    delta = torch.cat(ds) * imp.GRAD_SCALE_FACTOR

    raw = imp.collect_stats(model, dataloader, "cpu", strategy="dbf", curvature="kron", nkp_iters=2)
    L_ref, R_ref = imp.nkp_fit(x, delta, num_iters=2)
    L = raw["o_cov"]["0"]
    R = raw["i_cov"]["0"]
    assert torch.allclose(L / L.norm(), L_ref, atol=1e-4)
    assert torch.allclose(R / R.norm(), R_ref, atol=1e-4)


def test_partition_layers_respects_budget_and_order():
    model = _TinyMLP()
    layers = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    # layer "0": 8*(36+25)=488 B, layer "2": 8*(25+9)=272 B
    assert imp._partition_layers(layers, 10**9) == [["0", "2"]]
    assert imp._partition_layers(layers, 600) == [["0"], ["2"]]
    assert imp._partition_layers(layers, 100) == [["0"], ["2"]]  # oversized layers get their own group
    assert imp._partition_layers(layers, 760) == [["0", "2"]]


@pytest.mark.parametrize("strategy", ["online", "dbf"])
def test_grouped_device_accumulation_matches_streaming(monkeypatch, strategy):
    """One pass per layer group with on-device accumulators gives the same factors as the CPU-streaming path."""
    torch.manual_seed(6)
    dataloader = [torch.randn(2, 7, 6) for _ in range(3)]
    calls = {"n": 0}

    def counting_loop(*args, **kwargs):
        calls["n"] += 1
        _fake_loop(*args, **kwargs)

    monkeypatch.setattr(imp, "_run_calibration_loop", counting_loop)

    torch.manual_seed(7)
    model_a = _TinyMLP()
    ref = imp.collect_stats(model_a, dataloader, "cpu", strategy=strategy, curvature="kron", nkp_iters=2)
    assert calls["n"] == 2
    torch.manual_seed(7)
    model_b = _TinyMLP()
    got = imp.collect_stats(model_b, dataloader, "cpu", strategy=strategy, curvature="kron", nkp_iters=2,
                            gpu_budget_gb=600e-9)  # forces two layer groups
    assert calls["n"] == 2 + 2 * 2  # nkp_iters x groups
    for key in ("i_cov", "o_cov", "i_norm", "o_norm"):
        for name in ref[key]:
            assert torch.allclose(got[key][name], ref[key][name], atol=1e-5, rtol=1e-5), (key, name)


def test_register_stats_attaches_dense_non_persistent_buffers():
    model = _TinyMLP()
    stats = {
        "i_norm": {"0": torch.ones(6), "2": torch.ones(5)},
        "o_norm": {"0": torch.ones(5), "2": torch.ones(3)},
        "i_cov": {"0": torch.eye(6), "2": torch.eye(5)},
        "o_cov": {"0": torch.eye(5), "2": torch.eye(3)},
        "stats_device": "cpu",
    }
    imp.register_stats(model, stats)
    assert torch.equal(model[0].i_cov, torch.eye(6))
    assert torch.equal(model[2].o_cov, torch.eye(3))
    assert not any(k.endswith("_cov") for k in model.state_dict())


def test_collect_stats_rejects_unknown_curvature():
    with pytest.raises(ValueError):
        imp.collect_stats(_TinyMLP(), [], "cpu", curvature="banana")
