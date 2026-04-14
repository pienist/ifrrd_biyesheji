#!/usr/bin/env python3
"""Model quantization utilities."""

import torch
import torch.quantization


def quantize_dynamic(model):
    """Apply dynamic quantization to model."""
    raise NotImplementedError


def quantize_static(model, calibration_data):
    """Apply static quantization to model."""
    raise NotImplementedError


def prepare_qat(model):
    """Prepare model for Quantization-Aware Training."""
    raise NotImplementedError
