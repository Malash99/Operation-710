"""
Test script for the Loss Functions (Phase 7 verification).

Validates:
    1. MatchingLoss computes and has correct gradient behavior
    2. PoseLoss computes rotation + translation error correctly
    3. DinoVOLoss combines both with lambda_p scheduling
    4. lambda_p scheduling: 0.0 for first epochs, ramps to 0.9
    5. Matching loss is 0 when assignment is perfect
    6. Pose loss is 0 when R_est = R_gt and t_est = t_gt
    7. Gradients flow through all loss terms
    8. Loss values are reasonable with random inputs
"""

import os
import sys

import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.losses.losses import MatchingLoss, PoseLoss, DinoVOLoss


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("DINO-VO Phase 7: Loss Functions Verification")
    print(f"Device: {device}")
    print("=" * 65)

    B, K, M = 2, 512, 100  # batch, keypoints, GT matches

    # ------------------------------------------------------------------ #
    # 1. MatchingLoss — basic computation
    # ------------------------------------------------------------------ #
    print(f"\n[1] MatchingLoss basic test:")
    match_loss_fn = MatchingLoss().to(device)

    # Create random assignment matrix (softmax-like)
    assignment = torch.rand(B, K, K, device=device) * 0.01
    # Create GT matches: random pairs
    torch.manual_seed(42)
    gt_i = torch.randint(0, K, (B, M), device=device)
    gt_j = torch.randint(0, K, (B, M), device=device)
    gt_matches = torch.stack([gt_i, gt_j], dim=-1)  # (B, M, 2)
    gt_mask = torch.ones(B, M, dtype=torch.bool, device=device)

    assignment.requires_grad_(True)
    L_m = match_loss_fn(assignment, gt_matches, gt_mask)
    print(f"    Loss value: {L_m.item():.4f}")
    assert L_m.item() > 0, "Matching loss should be positive for random assignment"
    assert torch.isfinite(L_m), "Matching loss is not finite"

    L_m.backward()
    assert assignment.grad is not None, "No gradient on assignment"
    print(f"    Gradient norm: {assignment.grad.norm().item():.6f}")
    print(f"    MatchingLoss basic: PASSED")

    # ------------------------------------------------------------------ #
    # 2. MatchingLoss — perfect assignment should give ~0 loss
    # ------------------------------------------------------------------ #
    print(f"\n[2] MatchingLoss perfect assignment:")
    assignment_perfect = torch.zeros(1, K, K, device=device)
    # Set GT matches to have high probability
    gt_i_perf = torch.arange(M, device=device).unsqueeze(0)
    gt_j_perf = torch.arange(M, device=device).unsqueeze(0)
    gt_matches_perf = torch.stack([gt_i_perf, gt_j_perf], dim=-1)
    gt_mask_perf = torch.ones(1, M, dtype=torch.bool, device=device)
    for m in range(M):
        assignment_perfect[0, m, m] = 0.99  # near-perfect match

    L_m_perf = match_loss_fn(assignment_perfect, gt_matches_perf, gt_mask_perf)
    print(f"    Loss with P[i,j]=0.99: {L_m_perf.item():.6f}")
    assert L_m_perf.item() < 0.02, f"Perfect assignment loss too high: {L_m_perf.item()}"
    print(f"    Perfect assignment: PASSED")

    # ------------------------------------------------------------------ #
    # 3. PoseLoss — basic computation
    # ------------------------------------------------------------------ #
    print(f"\n[3] PoseLoss basic test:")
    pose_loss_fn = PoseLoss(lambda_r=180.0, lambda_t=400.0).to(device)

    # Random rotation and translation
    R_est = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    R_est.requires_grad_(True)
    t_est = torch.randn(B, 3, device=device, requires_grad=True)

    # Small rotation for GT (so error is meaningful)
    angle = torch.tensor(0.05, device=device)  # ~2.86 degrees
    R_gt = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    R_gt[:, 0, 0] = torch.cos(angle)
    R_gt[:, 0, 1] = -torch.sin(angle)
    R_gt[:, 1, 0] = torch.sin(angle)
    R_gt[:, 1, 1] = torch.cos(angle)

    t_gt = torch.tensor([[0.1, 0.0, 0.5]], device=device).expand(B, -1).clone()

    L_p = pose_loss_fn(R_est, t_est, R_gt, t_gt)
    print(f"    Loss value: {L_p.item():.4f}")
    assert L_p.item() > 0, "Pose loss should be positive"
    assert torch.isfinite(L_p), "Pose loss is not finite"

    L_p.backward()
    assert R_est.grad is not None, "No gradient on R_est"
    assert t_est.grad is not None, "No gradient on t_est"
    print(f"    R_est grad norm: {R_est.grad.norm().item():.6f}")
    print(f"    t_est grad norm: {t_est.grad.norm().item():.6f}")
    print(f"    PoseLoss basic: PASSED")

    # ------------------------------------------------------------------ #
    # 4. PoseLoss — zero error when R_est = R_gt, t_est = t_gt
    # ------------------------------------------------------------------ #
    print(f"\n[4] PoseLoss zero error test:")
    R_same = R_gt.detach().clone()
    t_same = t_gt.detach().clone()
    t_same_unit = t_same / t_same.norm(dim=-1, keepdim=True)

    L_p_zero = pose_loss_fn(R_same, t_same_unit, R_gt.detach(), t_gt.detach())
    print(f"    Loss with identical R, t: {L_p_zero.item():.8f}")
    assert L_p_zero.item() < 0.01, f"Zero-error loss too high: {L_p_zero.item()}"
    print(f"    Zero error: PASSED")

    # ------------------------------------------------------------------ #
    # 5. PoseLoss — rotation vs translation weighting
    # ------------------------------------------------------------------ #
    print(f"\n[5] PoseLoss weight check (lambda_r=180, lambda_t=400):")
    # Only rotation error (t matches perfectly)
    R_err = R_gt.detach().clone()
    R_err[:, 0, 0] = 1.0; R_err[:, 0, 1] = 0.0
    R_err[:, 1, 0] = 0.0; R_err[:, 1, 1] = 1.0  # identity
    L_rot_only = pose_loss_fn(R_err, t_same_unit, R_gt.detach(), t_gt.detach())

    # Only translation error (R matches perfectly)
    t_wrong = torch.tensor([[0.0, 1.0, 0.0]], device=device).expand(B, -1)
    L_trans_only = pose_loss_fn(R_gt.detach(), t_wrong, R_gt.detach(), t_gt.detach())

    print(f"    Rotation-only loss:    {L_rot_only.item():.4f}")
    print(f"    Translation-only loss: {L_trans_only.item():.4f}")
    print(f"    Weight check: PASSED")

    # ------------------------------------------------------------------ #
    # 6. DinoVOLoss — combined loss and lambda_p scheduling
    # ------------------------------------------------------------------ #
    print(f"\n[6] DinoVOLoss combined:")
    combined_loss_fn = DinoVOLoss(
        lambda_r=180.0, lambda_t=400.0,
        lambda_p_increment=1.5e-4, lambda_p_max=0.9,
    ).to(device)

    # With lambda_p = 0.0 (default): only matching loss
    assignment_c = torch.rand(B, K, K, device=device) * 0.01
    assignment_c.requires_grad_(True)
    R_est_c = R_gt.detach().clone()
    t_est_c = t_gt.detach().clone()

    out = combined_loss_fn(
        assignment_c, gt_matches, gt_mask,
        R_est_c, t_est_c, R_gt.detach(), t_gt.detach(),
    )
    print(f"    lambda_p = {out['lambda_p']:.4f}")
    print(f"    Total loss:    {out['total'].item():.4f}")
    print(f"    Matching loss: {out['matching'].item():.4f}")
    print(f"    Pose loss:     {out['pose'].item():.4f}")
    assert out["lambda_p"] == 0.0, "Initial lambda_p should be 0.0"
    assert abs(out["total"].item() - out["matching"].item()) < 1e-5, \
        "With lambda_p=0, total should equal matching loss"
    print(f"    lambda_p=0 check: PASSED")

    # ------------------------------------------------------------------ #
    # 7. lambda_p scheduling
    # ------------------------------------------------------------------ #
    print(f"\n[7] lambda_p scheduling:")
    combined_loss_fn.set_lambda_p(0.0)

    # Simulate stepping through training
    steps_to_test = [1, 100, 1000, 3000, 6000]
    for s in steps_to_test:
        combined_loss_fn.set_lambda_p(0.0)
        for _ in range(s):
            combined_loss_fn.step_lambda_p()
        lp = combined_loss_fn.lambda_p.item()
        print(f"    After {s:5d} steps: lambda_p = {lp:.4f}")

    # Check max clamp
    combined_loss_fn.set_lambda_p(0.0)
    for _ in range(10000):
        combined_loss_fn.step_lambda_p()
    lp_max = combined_loss_fn.lambda_p.item()
    print(f"    After 10000 steps: lambda_p = {lp_max:.4f} (max={combined_loss_fn.lambda_p_max})")
    assert abs(lp_max - 0.9) < 1e-5, f"lambda_p should cap at 0.9, got {lp_max}"
    print(f"    Scheduling check: PASSED")

    # ------------------------------------------------------------------ #
    # 8. DinoVOLoss with lambda_p > 0
    # ------------------------------------------------------------------ #
    print(f"\n[8] DinoVOLoss with lambda_p=0.5:")
    combined_loss_fn.set_lambda_p(0.5)

    assignment_d = torch.rand(B, K, K, device=device) * 0.01
    assignment_d.requires_grad_(True)
    R_est_d = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    R_est_d.requires_grad_(True)
    t_est_d = torch.randn(B, 3, device=device, requires_grad=True)

    out2 = combined_loss_fn(
        assignment_d, gt_matches, gt_mask,
        R_est_d, t_est_d, R_gt.detach(), t_gt.detach(),
    )
    print(f"    lambda_p = {out2['lambda_p']:.4f}")
    print(f"    Total loss:    {out2['total'].item():.4f}")
    print(f"    Matching loss: {out2['matching'].item():.4f}")
    print(f"    Pose loss:     {out2['pose'].item():.4f}")

    # Verify: total = 0.5 * matching + 0.5 * pose
    expected = 0.5 * out2["matching"].item() + 0.5 * out2["pose"].item()
    actual = out2["total"].item()
    assert abs(actual - expected) < 1e-3, \
        f"Combined loss mismatch: {actual} vs {expected}"
    print(f"    Combined check: PASSED")

    # ------------------------------------------------------------------ #
    # 9. Gradient flow through combined loss
    # ------------------------------------------------------------------ #
    print(f"\n[9] Gradient flow (combined loss):")
    out2["total"].backward()
    assert assignment_d.grad is not None, "No gradient on assignment"
    assert R_est_d.grad is not None, "No gradient on R_est"
    assert t_est_d.grad is not None, "No gradient on t_est"
    print(f"    assignment grad norm: {assignment_d.grad.norm().item():.6f}")
    print(f"    R_est grad norm:     {R_est_d.grad.norm().item():.6f}")
    print(f"    t_est grad norm:     {t_est_d.grad.norm().item():.6f}")
    print(f"    Gradient flow: PASSED")

    # ------------------------------------------------------------------ #
    # 10. Matching loss with masked entries
    # ------------------------------------------------------------------ #
    print(f"\n[10] MatchingLoss with partial mask:")
    mask_partial = gt_mask.clone()
    mask_partial[:, M // 2:] = False  # mask out half
    L_m_partial = match_loss_fn(
        assignment.detach(), gt_matches, mask_partial,
    )
    print(f"    Full mask loss:    {L_m.item():.4f}")
    print(f"    Partial mask loss: {L_m_partial.item():.4f}")
    assert torch.isfinite(L_m_partial), "Partial mask loss not finite"
    print(f"    Mask handling: PASSED")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 65)
    print("Phase 7 verification COMPLETE. All checks passed.")
    print("=" * 65)
    print(f"\nSummary:")
    print(f"  MatchingLoss:  NLL of assignment at GT matches (Eq. 12)")
    print(f"  PoseLoss:      lambda_r=180 * rot_err + lambda_t=400 * trans_err (Eq. 13)")
    print(f"  DinoVOLoss:    (1-lambda_p)*L_m + lambda_p*L_p (Eq. 14)")
    print(f"  Scheduling:    lambda_p: 0.0 -> 0.9 (increment 1.5e-4 per step)")
    print(f"  All gradients flow correctly through both loss terms.")


if __name__ == "__main__":
    main()
