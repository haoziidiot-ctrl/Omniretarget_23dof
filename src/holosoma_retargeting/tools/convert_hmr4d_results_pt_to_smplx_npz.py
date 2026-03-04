#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _require_package(name: str):
    try:
        return __import__(name)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Missing dependency '{name}'. Install it in your env, e.g. `pip install {name}` (or your project deps)."
        ) from e


def _as_tensor(x, torch):
    if torch.is_tensor(x):
        return x
    return torch.as_tensor(x)


def _get_smpl_params(root: dict) -> dict:
    if "smpl_params_global" in root and isinstance(root["smpl_params_global"], dict):
        return root["smpl_params_global"]
    if "pred_smpl_params_global" in root and isinstance(root["pred_smpl_params_global"], dict):
        return root["pred_smpl_params_global"]
    raise KeyError("Expected 'smpl_params_global' (or 'pred_smpl_params_global') dict in the loaded .pt file.")


def _broadcast_betas(betas, T: int, torch, mode: str):
    betas = _as_tensor(betas, torch).to(dtype=torch.float32, device="cpu")

    if betas.ndim == 1:
        betas = betas[None, :]

    if betas.shape[0] == 1:
        return betas.repeat(T, 1)

    if betas.shape[0] != T:
        raise ValueError(f"betas has shape {tuple(betas.shape)} but expected (T, B) with T={T} or (B,).")

    if mode == "per-frame":
        return betas
    if mode == "first":
        return betas[0:1].repeat(T, 1)
    if mode == "mean":
        return betas.mean(dim=0, keepdim=True).repeat(T, 1)
    raise ValueError(f"Unknown betas_mode: {mode}")


def _expected_body_pose_dim(model_type: str) -> int:
    # smpl: 23 joints * 3 (excludes global_orient) => 69
    # smplh/smplx: 21 joints * 3 (excludes global_orient) => 63
    if model_type == "smpl":
        return 69
    if model_type in {"smplh", "smplx"}:
        return 63
    raise ValueError(f"Unknown model_type: {model_type}")


def _maybe_pad_body_pose(body_pose, model_type: str, torch):
    """Pad/truncate body_pose axis-angle to match the expected parameterization.

    Common case: HMR pipelines sometimes output SMPL body_pose without the last 2 joints
    (L/R hand), i.e. 21*3=63 instead of 23*3=69. For model_type='smpl', we pad zeros.
    """
    if body_pose.ndim != 2 or body_pose.shape[1] % 3 != 0:
        raise ValueError(f"Expected body_pose shape (T, 3*K), got {tuple(body_pose.shape)}")

    target = _expected_body_pose_dim(model_type)
    cur = int(body_pose.shape[1])
    if cur == target:
        return body_pose
    if cur < target:
        pad = torch.zeros((body_pose.shape[0], target - cur), dtype=body_pose.dtype, device=body_pose.device)
        return torch.cat([body_pose, pad], dim=1)
    raise ValueError(f"body_pose dim too large for {model_type}: expected {target}, got {cur}")


def _compute_height_tpose(smpl_model, betas_TxB, torch, axis: str, model_type: str) -> float:
    T = int(betas_TxB.shape[0])
    device = torch.device("cpu")

    body_pose = torch.zeros((T, _expected_body_pose_dim(model_type)), dtype=torch.float32, device=device)
    global_orient = torch.zeros((T, 3), dtype=torch.float32, device=device)
    transl = torch.zeros((T, 3), dtype=torch.float32, device=device)

    out = smpl_model(betas=betas_TxB, body_pose=body_pose, global_orient=global_orient, transl=transl, pose2rot=True)
    if not hasattr(out, "vertices"):
        raise RuntimeError("SMPL model output has no 'vertices'; cannot compute height robustly.")

    verts = out.vertices.detach().cpu().numpy().reshape(-1, 3)
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_to_idx:
        raise ValueError(f"Invalid height_axis: {axis} (expected one of x/y/z)")
    idx = axis_to_idx[axis]
    return float(verts[:, idx].max() - verts[:, idx].min())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an HMR4D/GVHMR-style .pt (SMPL params) into a holosoma-compatible .npz "
            "with (T,J,3) `global_joint_positions` + scalar `height` for --data-format smplx."
        )
    )
    parser.add_argument("--pt", required=True, type=Path, help="Path to input .pt (e.g. demo_data/hmr4d_results.pt)")
    parser.add_argument("--out", required=True, type=Path, help="Path to output .npz (e.g. demo_data/gvhmr/seq1.npz)")
    parser.add_argument(
        "--smpl-model-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing SMPL model files. If omitted, uses $SMPL_MODEL_DIR. "
            "For smplx.create(model_type='smpl'), typical layout: $DIR/smpl/SMPL_NEUTRAL.pkl"
        ),
    )
    parser.add_argument("--model-type", default="smpl", choices=["smpl", "smplx", "smplh"], help="SMPL family type.")
    parser.add_argument("--gender", default="neutral", choices=["male", "female", "neutral"], help="Gender for model.")
    parser.add_argument(
        "--betas-mode",
        default="first",
        choices=["first", "per-frame", "mean"],
        help="How to handle per-frame betas (most pipelines expect betas constant).",
    )
    parser.add_argument(
        "--select-first-joints",
        type=int,
        default=22,
        help="Keep only the first N joints from the SMPL output (holosoma smplx expects 22 body joints).",
    )
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap y and z axes on output joints (i.e. [x,y,z]->[x,z,y]) if your input is y-up but you want z-up.",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        help=(
            "Negate the x axis on output joints (mirror). Useful to fix left/right being swapped after --swap-yz "
            "due to handedness changes."
        ),
    )
    parser.add_argument(
        "--height-axis",
        default="y",
        choices=["x", "y", "z"],
        help="Axis used to compute height from T-pose vertices (default y, common for SMPL).",
    )
    args = parser.parse_args()

    torch = _require_package("torch")
    np = _require_package("numpy")
    smplx = _require_package("smplx")

    pt_path = args.pt
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    root = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(root, dict):
        raise TypeError(f"Expected dict from torch.load, got {type(root)}")

    smpl_params = _get_smpl_params(root)
    required = ["body_pose", "betas", "global_orient", "transl"]
    missing = [k for k in required if k not in smpl_params]
    if missing:
        raise KeyError(f"Missing keys in smpl params: {missing}. Available: {list(smpl_params.keys())}")

    body_pose = _as_tensor(smpl_params["body_pose"], torch).to(dtype=torch.float32, device="cpu")
    global_orient = _as_tensor(smpl_params["global_orient"], torch).to(dtype=torch.float32, device="cpu")
    transl = _as_tensor(smpl_params["transl"], torch).to(dtype=torch.float32, device="cpu")

    if body_pose.ndim != 2 or body_pose.shape[1] % 3 != 0:
        raise ValueError(f"Expected body_pose shape (T, 3*K), got {tuple(body_pose.shape)}")
    if global_orient.shape != (body_pose.shape[0], 3):
        raise ValueError(f"Expected global_orient shape (T,3), got {tuple(global_orient.shape)}")
    if transl.shape != (body_pose.shape[0], 3):
        raise ValueError(f"Expected transl shape (T,3), got {tuple(transl.shape)}")

    T = int(body_pose.shape[0])
    betas_TxB = _broadcast_betas(smpl_params["betas"], T=T, torch=torch, mode=args.betas_mode)
    num_betas = int(betas_TxB.shape[1])

    model_dir = args.smpl_model_dir or Path(os.environ.get("SMPL_MODEL_DIR", ""))
    if not str(model_dir):
        raise RuntimeError("SMPL model dir not set. Pass --smpl-model-dir or set env var SMPL_MODEL_DIR.")
    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    smpl_model = smplx.create(
        str(model_dir),
        model_type=args.model_type,
        gender=args.gender,
        num_betas=num_betas,
        batch_size=T,
    )

    body_pose = _maybe_pad_body_pose(body_pose, model_type=args.model_type, torch=torch)

    out = smpl_model(
        betas=betas_TxB,
        body_pose=body_pose,
        global_orient=global_orient,
        transl=transl,
        pose2rot=True,
    )

    if not hasattr(out, "joints"):
        raise RuntimeError("SMPL model output has no 'joints'.")

    joints = out.joints.detach().cpu().numpy()
    if joints.ndim != 3 or joints.shape[0] != T or joints.shape[2] != 3:
        raise ValueError(f"Expected joints shape (T,J,3), got {tuple(joints.shape)}")

    keep_n = int(args.select_first_joints)
    if keep_n > 0 and joints.shape[1] >= keep_n:
        joints = joints[:, :keep_n, :]

    if args.swap_yz:
        joints = joints[..., [0, 2, 1]]

    if args.flip_x:
        joints[..., 0] *= -1.0

    height = _compute_height_tpose(smpl_model, betas_TxB, torch=torch, axis=args.height_axis, model_type=args.model_type)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(args.out), global_joint_positions=joints, height=float(height))
    print(f"Saved {args.out} with global_joint_positions={joints.shape}, height={height:.4f}")


if __name__ == "__main__":
    main()
