import re
from typing import Any


class GpuHardwareProfiler:
    """
    Universal GPU hardware profiler and architecture classifier for
    NVIDIA GeForce RTX 30-series (Ampere), 40-series (Ada Lovelace),
    and 50-series (Blackwell) graphic cards.
    """

    @classmethod
    def get_torch_compute_capability(cls) -> tuple[int, int] | None:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_capability(0)
        except Exception:
            pass
        return None

    @classmethod
    def classify_hardware(
        cls,
        gpu_name: str | None,
        total_vram_mb: int = 0,
        compute_cap: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        raw_name = (gpu_name or "").strip()
        name_lower = raw_name.lower()

        if compute_cap is None:
            compute_cap = cls.get_torch_compute_capability()

        # Architecture detection
        arch_name = "Generic"
        series_name = "NVIDIA GeForce RTX"
        cap_str: str | None = None
        cap_version: str | None = None

        if compute_cap is not None:
            major, minor = compute_cap
            cap_version = f"{major}.{minor}"
            cap_str = f"sm_{major}{minor}"
            if major >= 12 or major == 10:  # Blackwell sm_120 / sm_100
                arch_name = "Blackwell"
                series_name = "RTX 50-Series (Blackwell)"
            elif major == 8 and minor == 9:  # Ada Lovelace sm_89
                arch_name = "Ada Lovelace"
                series_name = "RTX 40-Series (Ada Lovelace)"
            elif major == 8 and (minor == 6 or minor == 0):  # Ampere sm_86 / sm_80
                arch_name = "Ampere"
                series_name = "RTX 30-Series (Ampere)"
            elif major == 7 and minor == 5:  # Turing sm_75
                arch_name = "Turing"
                series_name = "RTX 20-Series (Turing)"
            elif major >= 8:
                arch_name = "Ampere Compatible"
                series_name = "NVIDIA Modern RTX"

        # Fallback / heuristic regex classification if compute_cap couldn't be queried directly
        if arch_name == "Generic":
            if re.search(r"rtx\s*50\d{2}", name_lower) or "blackwell" in name_lower:
                arch_name = "Blackwell"
                series_name = "RTX 50-Series (Blackwell)"
                cap_str = "sm_120"
                cap_version = "12.0"
            elif re.search(r"rtx\s*40\d{2}", name_lower) or "ada" in name_lower:
                arch_name = "Ada Lovelace"
                series_name = "RTX 40-Series (Ada Lovelace)"
                cap_str = "sm_89"
                cap_version = "8.9"
            elif re.search(r"rtx\s*30\d{2}", name_lower) or "ampere" in name_lower or "a100" in name_lower:
                arch_name = "Ampere"
                series_name = "RTX 30-Series (Ampere)"
                cap_str = "sm_86"
                cap_version = "8.6"
            elif re.search(r"rtx\s*20\d{2}", name_lower) or "turing" in name_lower or "titan rtx" in name_lower:
                arch_name = "Turing"
                series_name = "RTX 20-Series (Turing)"
                cap_str = "sm_75"
                cap_version = "7.5"

        # VRAM Tiering
        if total_vram_mb <= 8500:
            vram_tier = "tier1_compact_8gb"
            tier_label = "8 GB Compact Tier"
            recommended_res = 512
            recommended_batch_size = 1
            copilot_headroom_mb = 3000
            notes_tier = "Compact memory ceiling. Recommended: 512–768px resolutions and batch size 1."
        elif total_vram_mb <= 13000:
            vram_tier = "tier2_mid_12gb"
            tier_label = "12 GB Mid Tier"
            recommended_res = 1024
            recommended_batch_size = 1
            copilot_headroom_mb = 3500
            notes_tier = "Balanced memory ceiling. Optimal for native 1024x1024 SDXL sampling."
        elif total_vram_mb <= 18000:
            vram_tier = "tier3_high_16gb"
            tier_label = "16 GB High Tier"
            recommended_res = 1024
            recommended_batch_size = 1
            copilot_headroom_mb = 4096
            notes_tier = "High memory headroom. Supports high-res SDXL and multi-LoRA stacking."
        else:
            vram_tier = "tier4_enthusiast_24gb"
            tier_label = "24 GB+ Enthusiast Tier"
            recommended_res = 1024
            recommended_batch_size = 2
            copilot_headroom_mb = 4096
            notes_tier = "Enthusiast workstation capacity. Concurrently runs rendering, LoRAs, and Copilot."

        # Precision support
        if arch_name in ("Blackwell", "Ada Lovelace"):
            precision_support = ["fp16", "bf16", "fp8", "tf32"]
            preferred_training_precision = "bf16"
        elif arch_name == "Ampere":
            precision_support = ["fp16", "bf16", "tf32"]
            preferred_training_precision = "bf16"
        else:
            precision_support = ["fp16"]
            preferred_training_precision = "fp16"

        # PyTorch installation / diagnostic remediation command recommendation
        if arch_name == "Blackwell":
            recommended_pytorch_cmd = "pip install torch torchvision --pre --index-url https://download.pytorch.org/whl/nightly/cu128"
            cuda_target_version = "12.8+"

        elif arch_name == "Ada Lovelace":
            recommended_pytorch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            cuda_target_version = "12.4"
        elif arch_name == "Ampere":
            recommended_pytorch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
            cuda_target_version = "12.1 / 12.4"
        else:
            recommended_pytorch_cmd = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            cuda_target_version = "12.1"

        return {
            "gpu_name": raw_name or "Unknown NVIDIA GPU",
            "gpu_series": series_name,
            "architecture": arch_name,
            "compute_capability": cap_str,
            "compute_capability_version": cap_version,
            "vram_tier": vram_tier,
            "vram_tier_label": tier_label,
            "total_vram_mb": total_vram_mb,
            "precision_support": precision_support,
            "preferred_training_precision": preferred_training_precision,
            "recommended_resolution": recommended_res,
            "recommended_batch_size": recommended_batch_size,
            "copilot_headroom_mb": copilot_headroom_mb,
            "recommended_pytorch_cmd": recommended_pytorch_cmd,
            "cuda_target_version": cuda_target_version,
            "notes": notes_tier,
        }

    @classmethod
    def get_adaptive_training_preset(
        cls,
        gpu_name: str | None,
        total_vram_mb: int = 0,
        compute_cap: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        profile = cls.classify_hardware(gpu_name, total_vram_mb, compute_cap)
        tier = profile["vram_tier"]
        arch = profile["architecture"]
        precision = profile["preferred_training_precision"]

        if tier == "tier1_compact_8gb":
            return {
                "resolution": 512,
                "batch_size": 1,
                "network_dim": 8,
                "network_alpha": 8,
                "learning_rate": 0.0001,
                "text_encoder_lr": 0.00005,
                "mixed_precision": precision,
                "gradient_checkpointing": True,
                "max_train_steps": 1000,
                "estimated_vram_gb": 6.8,
                "gpu_architecture": arch,
                "gpu_series": profile["gpu_series"],
                "vram_tier": tier,
                "notes": f"Auto-tuned for {profile['gpu_series']} ({profile['vram_tier_label']}) - 512px with gradient checkpointing.",
            }
        elif tier == "tier2_mid_12gb":
            return {
                "resolution": 1024,
                "batch_size": 1,
                "network_dim": 16,
                "network_alpha": 16,
                "learning_rate": 0.0001,
                "text_encoder_lr": 0.00005,
                "mixed_precision": precision,
                "gradient_checkpointing": True,
                "max_train_steps": 1000,
                "estimated_vram_gb": 10.5,
                "gpu_architecture": arch,
                "gpu_series": profile["gpu_series"],
                "vram_tier": tier,
                "notes": f"Auto-tuned for {profile['gpu_series']} ({profile['vram_tier_label']}) - SDXL 1024px balanced memory.",
            }
        elif tier == "tier3_high_16gb":
            return {
                "resolution": 1024,
                "batch_size": 1,
                "network_dim": 16,
                "network_alpha": 16,
                "learning_rate": 0.0001,
                "text_encoder_lr": 0.00005,
                "mixed_precision": precision,
                "gradient_checkpointing": True,
                "max_train_steps": 1000,
                "estimated_vram_gb": 12.5,
                "gpu_architecture": arch,
                "gpu_series": profile["gpu_series"],
                "vram_tier": tier,
                "notes": f"Auto-tuned for {profile['gpu_series']} ({profile['vram_tier_label']}) - High fidelity rank 16 with {precision}.",
            }
        else:  # tier4_enthusiast_24gb
            return {
                "resolution": 1024,
                "batch_size": 2,
                "network_dim": 32,
                "network_alpha": 32,
                "learning_rate": 0.0001,
                "text_encoder_lr": 0.00005,
                "mixed_precision": precision,
                "gradient_checkpointing": False,
                "max_train_steps": 1000,
                "estimated_vram_gb": 18.0,
                "gpu_architecture": arch,
                "gpu_series": profile["gpu_series"],
                "vram_tier": tier,
                "notes": f"Auto-tuned for {profile['gpu_series']} ({profile['vram_tier_label']}) - Enthusiast batch size 2, rank 32.",
            }

