import pytest
from app.services.gpu_profiler import GpuHardwareProfiler
from app.services.gpu_safety import GpuSafetyManager
from app.schemas.system import TelemetryResponse


def test_rtx_30_series_classification():
    # Test RTX 3060 12GB
    p3060 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 3060", total_vram_mb=12288, compute_cap=(8, 6))
    assert p3060["architecture"] == "Ampere"
    assert "RTX 30-Series" in p3060["gpu_series"]
    assert p3060["compute_capability"] == "sm_86"
    assert p3060["vram_tier"] == "tier2_mid_12gb"
    assert "bf16" in p3060["precision_support"]
    assert "fp8" not in p3060["precision_support"]  # Ampere lacks FP8 tensor cores

    # Test RTX 3070 8GB
    p3070 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 3070", total_vram_mb=8192, compute_cap=(8, 6))
    assert p3070["vram_tier"] == "tier1_compact_8gb"
    assert p3070["recommended_resolution"] == 512

    # Test RTX 3090 24GB
    p3090 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 3090", total_vram_mb=24576, compute_cap=(8, 6))
    assert p3090["vram_tier"] == "tier4_enthusiast_24gb"
    assert p3090["recommended_batch_size"] == 2


def test_rtx_40_series_classification():
    # Test RTX 4060 8GB
    p4060 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 4060", total_vram_mb=8192, compute_cap=(8, 9))
    assert p4060["architecture"] == "Ada Lovelace"
    assert "RTX 40-Series" in p4060["gpu_series"]
    assert p4060["compute_capability"] == "sm_89"
    assert p4060["vram_tier"] == "tier1_compact_8gb"
    assert "fp8" in p4060["precision_support"]

    # Test RTX 4070 12GB
    p4070 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 4070", total_vram_mb=12288, compute_cap=(8, 9))
    assert p4070["vram_tier"] == "tier2_mid_12gb"

    # Test RTX 4080 16GB
    p4080 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 4080", total_vram_mb=16384, compute_cap=(8, 9))
    assert p4080["vram_tier"] == "tier3_high_16gb"

    # Test RTX 4090 24GB
    p4090 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 4090", total_vram_mb=24576, compute_cap=(8, 9))
    assert p4090["vram_tier"] == "tier4_enthusiast_24gb"
    assert "fp8" in p4090["precision_support"]


def test_rtx_50_series_classification():
    # Test RTX 5070 12GB
    p5070 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 5070", total_vram_mb=12288, compute_cap=(12, 0))
    assert p5070["architecture"] == "Blackwell"
    assert "RTX 50-Series" in p5070["gpu_series"]
    assert p5070["compute_capability"] == "sm_120"
    assert p5070["vram_tier"] == "tier2_mid_12gb"
    assert "fp8" in p5070["precision_support"]

    # Test RTX 5070 Ti 16GB
    p5070ti = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 5070 Ti", total_vram_mb=16384, compute_cap=(12, 0))
    assert p5070ti["architecture"] == "Blackwell"
    assert p5070ti["vram_tier"] == "tier3_high_16gb"
    assert "cu128" in p5070ti["recommended_pytorch_cmd"]

    # Test RTX 5090 32GB
    p5090 = GpuHardwareProfiler.classify_hardware("NVIDIA GeForce RTX 5090", total_vram_mb=32768, compute_cap=(12, 0))
    assert p5090["vram_tier"] == "tier4_enthusiast_24gb"


def test_adaptive_training_presets():
    # Preset for 8GB Ampere (RTX 3070)
    preset_8gb = GpuHardwareProfiler.get_adaptive_training_preset("NVIDIA GeForce RTX 3070", 8192, (8, 6))
    assert preset_8gb["resolution"] == 512
    assert preset_8gb["batch_size"] == 1
    assert preset_8gb["gradient_checkpointing"] is True
    assert preset_8gb["estimated_vram_gb"] <= 7.5

    # Preset for 12GB Ada (RTX 4070)
    preset_12gb = GpuHardwareProfiler.get_adaptive_training_preset("NVIDIA GeForce RTX 4070", 12288, (8, 9))
    assert preset_12gb["resolution"] == 1024
    assert preset_12gb["batch_size"] == 1
    assert preset_12gb["network_dim"] == 16

    # Preset for 16GB Blackwell (RTX 5070 Ti)
    preset_16gb = GpuHardwareProfiler.get_adaptive_training_preset("NVIDIA GeForce RTX 5070 Ti", 16384, (12, 0))
    assert preset_16gb["resolution"] == 1024
    assert preset_16gb["mixed_precision"] == "bf16"
    assert preset_16gb["network_dim"] == 16

    # Preset for 24GB+ Enthusiast (RTX 3090 / 4090 / 5090)
    preset_24gb = GpuHardwareProfiler.get_adaptive_training_preset("NVIDIA GeForce RTX 4090", 24576, (8, 9))
    assert preset_24gb["batch_size"] == 2
    assert preset_24gb["network_dim"] == 32


def test_copilot_headroom_tier_awareness():
    # On 8GB card (3070), Copilot requires 3000 MB headroom
    allowed, msg = GpuSafetyManager.can_run_copilot(
        policy="balanced",
        free_vram_mb=3200,
        is_heavy_render_active=False,
        total_vram_mb=8192,
        gpu_name="NVIDIA GeForce RTX 3070",
    )
    assert allowed is True

    # On 8GB card, Copilot is blocked during active render
    blocked_render, msg2 = GpuSafetyManager.can_run_copilot(
        policy="balanced",
        free_vram_mb=4000,
        is_heavy_render_active=True,
        total_vram_mb=8192,
        gpu_name="NVIDIA GeForce RTX 3070",
    )
    assert blocked_render is False
    assert "8GB tier" in msg2

    # On 16GB card, 3200 MB free is insufficient under balanced policy (requires 4096 MB)
    allowed_16, msg16 = GpuSafetyManager.can_run_copilot(
        policy="balanced",
        free_vram_mb=3200,
        is_heavy_render_active=False,
        total_vram_mb=16384,
        gpu_name="NVIDIA GeForce RTX 5070 Ti",
    )
    assert allowed_16 is False
