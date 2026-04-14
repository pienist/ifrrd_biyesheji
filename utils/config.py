#!/usr/bin/env python3
"""Configuration management for YOLOv11-ConvNeXt."""

import yaml
from pathlib import Path


class Config:
    """Configuration handler."""
    
    def __init__(self, config_path=None):
        self.config_path = config_path
        self.data = {}
        
    def load(self, path=None):
        """Load configuration from YAML file."""
        path = path or self.config_path
        with open(path, 'r') as f:
            self.data = yaml.safe_load(f)
        return self.data
    
    def save(self, path):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.data, f)
            
    def get(self, key, default=None):
        """Get configuration value."""
        return self.data.get(key, default)
