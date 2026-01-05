#!/usr/bin/env python3
"""Standalone script to download, convert, and package LoRA adapter weights for Kaggle submission.

Usage:
    python3 convert_and_package.py
    python3 convert_and_package.py --tinker-path tinker://your-model-path
    python3 convert_and_package.py --local-dir ./weights
"""

import argparse
import json
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def get_tinker_key():
    """Retrieve Tinker API Key from environment or env.json."""
    if "TINKER_API_KEY" in os.environ:
        return os.environ["TINKER_API_KEY"]
    if os.path.exists("env.json"):
        with open("env.json") as f:
            try:
                data = json.load(f)
                return data.get("TINKER_API_KEY")
            except Exception:
                pass
    return None


def find_latest_adapter():
    """Find the latest sampler_weights/final checkpoint via tinker SDK."""
    import tinker

    tinker_key = get_tinker_key()
    if tinker_key:
        os.environ["TINKER_API_KEY"] = tinker_key
    else:
        print("Warning: TINKER_API_KEY not found in environment or env.json.")

    print("Connecting to Tinker to find the latest checkpoint...")
    sc = tinker.ServiceClient()
    checkpoints = sc.create_rest_client().list_checkpoints().result().checkpoints

    candidates = [c for c in checkpoints if c.checkpoint_id == "sampler_weights/final"]
    if not candidates:
        raise ValueError("No sampler_weights/final checkpoint found in Tinker")

    # Select the max by time
    latest = max(candidates, key=lambda c: c.time)
    print(f"Found latest checkpoint: {latest.tinker_path} (created {latest.time})")
    return latest.tinker_path


def download_adapter_from_tinker(tinker_path, download_dir):
    """Download adapter weights from Tinker to a local directory."""
    import tinker

    tinker_key = get_tinker_key()
    if tinker_key:
        os.environ["TINKER_API_KEY"] = tinker_key

    print(f"Downloading checkpoint archive from Tinker: {tinker_path}...")
    os.makedirs(download_dir, exist_ok=True)

    model_id_match = re.search(r"tinker://([a-f0-9-]+)", tinker_path)
    model_id_str = model_id_match.group(1) if model_id_match else "unknown"

    sc = tinker.ServiceClient()
    url = (
        sc.create_rest_client()
        .get_checkpoint_archive_url_from_tinker_path(tinker_path)
        .result()
        .url
    )

    tar_path = os.path.join(tempfile.gettempdir(), f"adapter_{model_id_str}.tar")
    print("Obtained archive download URL. Downloading tar file...")
    urllib.request.urlretrieve(url, tar_path)
    print(f"Download complete: {os.path.getsize(tar_path) / 1e6:.1f} MB")

    print(f"Extracting archive to {download_dir}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(download_dir)
    os.remove(tar_path)

    # Flatten structure if files extracted to a subdirectory
    config_path = None
    weights_path = None
    for root, _dirs, files in os.walk(download_dir):
        for f in files:
            if f == "adapter_config.json":
                config_path = os.path.join(root, f)
            elif f == "adapter_model.safetensors":
                weights_path = os.path.join(root, f)

    if config_path and weights_path:
        for path in [config_path, weights_path]:
            dest = os.path.join(download_dir, os.path.basename(path))
            if path != dest:
                shutil.move(path, dest)


def trained_adapter_key_rename(key_name: str) -> str:
    """Rename weights base model suffix to match submission requirements."""
    return key_name.replace("base_model.model.model", "base_model.model.backbone")


def convert_and_package_adapter(adapter_dir, output_zip):
    """Align config, convert tensors, and zip adapter."""
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_dir}")

    # 1. Patch target modules in adapter_config.json
    print("Patching adapter_config.json...")
    with open(config_path, "r") as f:
        config_data = json.load(f)

    config_data["target_modules"] = [
        "k_proj",
        "o_proj",
        "in_proj",
        "q_proj",
        "up_proj",
        "v_proj",
        "down_proj",
        "out_proj",
        "lm_head",
    ]

    with open(config_path, "w") as f:
        json.dump(config_data, f)
    print("Patched target_modules in adapter_config.json successfully.")

    # 2. Run tensor conversion (rename, expert unfusing, Mamba QR/SVD merge)
    print("Running tensor conversion...")
    adapter_tensors = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            adapter_tensors[key] = f.get_tensor(key)

    # Collect adapter base names
    base_names = set()
    for key in adapter_tensors:
        base = re.sub(r"\.lora_[AB]\.weight$", "", key)
        base_names.add(base)

    # Identify Mamba layers needing gate_proj+x_proj -> in_proj
    mamba_merge_layers = {}
    for base in base_names:
        for proj in ("gate_proj", "x_proj"):
            if f".{proj}" in base:
                layer_path = base.rsplit(f".{proj}", 1)[0]
                mamba_merge_layers.setdefault(layer_path, {})[proj] = base

    mamba_merge_bases = set()
    for projs in mamba_merge_layers.values():
        mamba_merge_bases.update(projs.values())

    # Build output tensors
    tensors = {}

    for base in sorted(base_names):
        lora_A = adapter_tensors[f"{base}.lora_A.weight"]
        lora_B = adapter_tensors[f"{base}.lora_B.weight"]
        renamed = trained_adapter_key_rename(base)

        # Skip empty w3 experts
        if ".experts.w3" in base and lora_A.numel() == 0:
            continue

        # Skip gate_proj/x_proj - handled in Mamba merge pass below
        if base in mamba_merge_bases:
            continue

        # Expert unfusing: w1 -> per-expert up_proj, w2 -> per-expert down_proj
        if ".experts.w1" in base or ".experts.w2" in base:
            if lora_A.shape[0] == 1:
                lora_A = lora_A.expand(lora_B.shape[0], -1, -1).contiguous()
            elif lora_B.shape[0] == 1:
                lora_B = lora_B.expand(lora_A.shape[0], -1, -1).contiguous()

            num_experts = lora_A.shape[0]
            proj_name = "up_proj" if ".w1" in base else "down_proj"

            for i in range(num_experts):
                exp_renamed = re.sub(
                    r"\.experts\.w[12]",
                    f".experts.{i}.{proj_name}",
                    renamed,
                )
                tensors[f"{exp_renamed}.lora_A.weight"] = lora_A[i].contiguous()
                tensors[f"{exp_renamed}.lora_B.weight"] = lora_B[i].contiguous()
            continue

        # Direct rename for everything else
        tensors[f"{renamed}.lora_A.weight"] = lora_A
        tensors[f"{renamed}.lora_B.weight"] = lora_B

    # Mamba: gate_proj + x_proj -> in_proj via SVD
    for layer_path, projs in sorted(mamba_merge_layers.items()):
        renamed_layer = trained_adapter_key_rename(layer_path)
        in_proj_base = f"{renamed_layer}.in_proj"

        gate_A = adapter_tensors[f"{projs['gate_proj']}.lora_A.weight"].float()
        gate_B = adapter_tensors[f"{projs['gate_proj']}.lora_B.weight"].float()
        x_A = adapter_tensors[f"{projs['x_proj']}.lora_A.weight"].float()
        x_B = adapter_tensors[f"{projs['x_proj']}.lora_B.weight"].float()
        rank = gate_A.shape[0]

        in_proj_dim = gate_B.shape[0] + x_B.shape[0]

        # Build combined rank-64 representation, then SVD to best rank-32
        A_cat = torch.cat([gate_A, x_A], dim=0)
        B_block = torch.zeros(in_proj_dim, 2 * rank)
        B_block[: gate_B.shape[0], :rank] = gate_B
        B_block[gate_B.shape[0] : gate_B.shape[0] + x_B.shape[0], rank:] = x_B

        Q_B, R_B = torch.linalg.qr(B_block)
        Q_A, R_A = torch.linalg.qr(A_cat.T)
        core = R_B @ R_A.T
        U, S, Vh = torch.linalg.svd(core, full_matrices=False)

        k = rank
        new_B = (Q_B @ U[:, :k]) * S[:k].unsqueeze(0)
        new_A = Vh[:k, :] @ Q_A.T

        kept = S[:k].sum().item()
        total = S.sum().item()
        print(
            f"{layer_path}: SVD kept {kept:.2f}/{total:.2f} "
            f"({kept / total * 100:.1f}%) of singular value mass"
        )

        tensors[f"{in_proj_base}.lora_A.weight"] = new_A
        tensors[f"{in_proj_base}.lora_B.weight"] = new_B

    print(
        f"Converted {len(adapter_tensors)} trained tensors -> {len(tensors)} output tensors"
    )

    # Save output tensors back to weights file
    save_file(tensors, weights_path)

    # 3. Create submission.zip containing only config and weights in the root
    print(f"Creating submission package at {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(config_path, "adapter_config.json")
        zf.write(weights_path, "adapter_model.safetensors")

    print(
        f"Submission package created successfully: {output_zip} ({os.path.getsize(output_zip) / 1e6:.2f} MB)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert and package adapter weights for Kaggle."
    )
    parser.add_argument("--tinker-path", help="Tinker checkpoint path (tinker://...)")
    parser.add_argument(
        "--local-dir", help="Use pre-downloaded local weights directory"
    )
    parser.add_argument(
        "--output", default="submission.zip", help="Output zip filename"
    )
    args = parser.parse_args()

    temp_dir = None
    try:
        if args.local_dir:
            adapter_dir = args.local_dir
            print(f"Using local adapter directory: {adapter_dir}")
        else:
            tinker_path = args.tinker_path
            if not tinker_path:
                tinker_path = find_latest_adapter()

            temp_dir = tempfile.mkdtemp()
            download_adapter_from_tinker(tinker_path, temp_dir)
            adapter_dir = temp_dir

        convert_and_package_adapter(adapter_dir, args.output)

    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
