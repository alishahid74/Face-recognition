"""
Plugin System for Custom Models
Allows users to add custom models dynamically
"""

import os
import sys
import importlib
import inspect
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class ModelPlugin(ABC):
    """Base class for model plugins"""
    
    @abstractmethod
    def load_model(self):
        """Load the model"""
        pass
    
    @abstractmethod
    def predict(self, input_data):
        """Run prediction"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Get model information"""
        pass
    
    def preprocess(self, input_data):
        """Preprocess input (optional)"""
        return input_data
    
    def postprocess(self, output_data):
        """Postprocess output (optional)"""
        return output_data


class CustomModelConfig:
    """Configuration for custom models"""
    
    def __init__(self, config_path: str):
        """
        Load model configuration from YAML
        
        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    @property
    def model_name(self) -> str:
        return self.config.get('name', 'Unknown')
    
    @property
    def model_type(self) -> str:
        return self.config.get('type', 'custom')
    
    @property
    def framework(self) -> str:
        return self.config.get('framework', 'unknown')
    
    @property
    def model_path(self) -> str:
        return self.config.get('model_path', '')
    
    @property
    def input_size(self) -> tuple:
        size = self.config.get('input_size', [224, 224])
        return tuple(size)
    
    @property
    def classes(self) -> List[str]:
        return self.config.get('classes', [])
    
    @property
    def preprocessing(self) -> Dict:
        return self.config.get('preprocessing', {})
    
    @property
    def metadata(self) -> Dict:
        return self.config.get('metadata', {})


class PluginManager:
    """Manage custom model plugins"""
    
    def __init__(self, plugin_dir: str = "plugins/models"):
        """
        Initialize plugin manager
        
        Args:
            plugin_dir: Directory containing plugin files
        """
        self.plugin_dir = Path(plugin_dir)
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins = {}
        self.configs = {}
        
        # Auto-discover plugins
        self.discover_plugins()
    
    def discover_plugins(self):
        """Automatically discover and load plugins"""
        print(f"Discovering plugins in {self.plugin_dir}...")
        
        # Add plugin directory to path
        sys.path.insert(0, str(self.plugin_dir))
        
        # Look for Python files
        for py_file in self.plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                module_name = py_file.stem
                module = importlib.import_module(module_name)
                
                # Find plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, ModelPlugin) and obj != ModelPlugin:
                        plugin_name = name
                        self.plugins[plugin_name] = obj
                        print(f"  ✓ Loaded plugin: {plugin_name}")
            
            except Exception as e:
                print(f"  ✗ Failed to load {py_file.name}: {e}")
        
        # Look for YAML configs
        for yaml_file in self.plugin_dir.glob("*.yaml"):
            try:
                config = CustomModelConfig(str(yaml_file))
                self.configs[config.model_name] = config
                print(f"  ✓ Loaded config: {config.model_name}")
            
            except Exception as e:
                print(f"  ✗ Failed to load {yaml_file.name}: {e}")
        
        print(f"Total plugins loaded: {len(self.plugins)}")
        print(f"Total configs loaded: {len(self.configs)}")
    
    def get_plugin(self, plugin_name: str) -> Optional[ModelPlugin]:
        """
        Get a plugin instance
        
        Args:
            plugin_name: Name of the plugin
        
        Returns:
            Plugin instance or None
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]()
        return None
    
    def get_config(self, config_name: str) -> Optional[CustomModelConfig]:
        """Get a model config"""
        return self.configs.get(config_name)
    
    def list_plugins(self) -> List[str]:
        """List all available plugins"""
        return list(self.plugins.keys())
    
    def list_configs(self) -> List[str]:
        """List all available configs"""
        return list(self.configs.keys())
    
    def create_plugin_template(self, plugin_name: str):
        """
        Create a template plugin file
        
        Args:
            plugin_name: Name for the new plugin
        """
        template = f'''"""
Custom Model Plugin: {plugin_name}
"""

from plugins.plugin_system import ModelPlugin
import torch
import numpy as np


class {plugin_name}(ModelPlugin):
    """Custom model plugin"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
    
    def load_model(self):
        """Load your custom model"""
        # TODO: Implement model loading
        # Example:
        # self.model = torch.load('path/to/model.pth')
        # self.model.to(self.device)
        # self.model.eval()
        pass
    
    def predict(self, input_data):
        """Run prediction"""
        # TODO: Implement prediction
        # Example:
        # preprocessed = self.preprocess(input_data)
        # with torch.no_grad():
        #     output = self.model(preprocessed)
        # return self.postprocess(output)
        pass
    
    def preprocess(self, input_data):
        """Preprocess input"""
        # TODO: Implement preprocessing
        return input_data
    
    def postprocess(self, output_data):
        """Postprocess output"""
        # TODO: Implement postprocessing
        return output_data
    
    def get_info(self):
        """Get model information"""
        return {{
            'name': '{plugin_name}',
            'type': 'custom',
            'framework': 'pytorch',
            'description': 'Custom model plugin'
        }}
'''
        
        output_path = self.plugin_dir / f"{plugin_name.lower()}.py"
        with open(output_path, 'w') as f:
            f.write(template)
        
        print(f"Plugin template created: {output_path}")
    
    def create_config_template(self, model_name: str):
        """
        Create a template YAML config
        
        Args:
            model_name: Name for the model config
        """
        template = f'''# Custom Model Configuration
name: {model_name}
type: classification  # or detection, segmentation, etc.
framework: pytorch  # pytorch, tensorflow, onnx, etc.

# Model file path
model_path: models/weights/{model_name.lower()}.pth

# Input configuration
input_size: [224, 224]
input_channels: 3

# Classes (for classification/detection)
classes:
  - class1
  - class2
  - class3

# Preprocessing configuration
preprocessing:
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  resize_mode: center_crop

# Inference configuration
inference:
  batch_size: 1
  device: auto  # auto, cpu, cuda
  confidence_threshold: 0.5

# Metadata
metadata:
  author: Your Name
  description: Custom model for specific task
  version: 1.0.0
  date: 2025-01-01
'''
        
        output_path = self.plugin_dir / f"{model_name.lower()}.yaml"
        with open(output_path, 'w') as f:
            f.write(template)
        
        print(f"Config template created: {output_path}")


class ONNXPlugin(ModelPlugin):
    """Plugin for ONNX models"""
    
    def __init__(self, model_path: str, config: Optional[CustomModelConfig] = None):
        """
        Initialize ONNX plugin
        
        Args:
            model_path: Path to ONNX model file
            config: Optional model configuration
        """
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        self.model_path = model_path
        self.config = config
        self.session = None
        self.load_model()
    
    def load_model(self):
        """Load ONNX model"""
        print(f"Loading ONNX model: {self.model_path}")
        self.session = self.ort.InferenceSession(self.model_path)
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print("✓ ONNX model loaded")
    
    def predict(self, input_data):
        """Run ONNX inference"""
        preprocessed = self.preprocess(input_data)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: preprocessed}
        )
        
        return self.postprocess(outputs[0])
    
    def get_info(self):
        """Get model information"""
        return {
            'name': Path(self.model_path).stem,
            'type': 'onnx',
            'framework': 'onnx',
            'path': self.model_path,
            'input_name': self.input_name,
            'output_name': self.output_name
        }


class TorchScriptPlugin(ModelPlugin):
    """Plugin for TorchScript models"""
    
    def __init__(self, model_path: str, config: Optional[CustomModelConfig] = None):
        """
        Initialize TorchScript plugin
        
        Args:
            model_path: Path to TorchScript model file
            config: Optional model configuration
        """
        import torch
        
        self.torch = torch
        self.model_path = model_path
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load TorchScript model"""
        print(f"Loading TorchScript model: {self.model_path}")
        self.model = self.torch.jit.load(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ TorchScript model loaded on {self.device}")
    
    def predict(self, input_data):
        """Run TorchScript inference"""
        preprocessed = self.preprocess(input_data)
        
        with self.torch.no_grad():
            output = self.model(preprocessed)
        
        return self.postprocess(output)
    
    def get_info(self):
        """Get model information"""
        return {
            'name': Path(self.model_path).stem,
            'type': 'torchscript',
            'framework': 'pytorch',
            'path': self.model_path,
            'device': str(self.device)
        }


# Example plugin implementation
class ExampleCustomPlugin(ModelPlugin):
    """Example custom plugin implementation"""
    
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load example model"""
        print("Loading example model...")
        # Placeholder - implement actual loading
        self.model = lambda x: {"prediction": "example_class", "confidence": 0.95}
    
    def predict(self, input_data):
        """Run example prediction"""
        return self.model(input_data)
    
    def get_info(self):
        """Get example model info"""
        return {
            'name': 'ExampleModel',
            'type': 'example',
            'framework': 'custom',
            'description': 'Example custom model plugin'
        }
