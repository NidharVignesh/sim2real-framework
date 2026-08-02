from pathlib import Path
import numpy as np

def load_sb3_policy(model_path: str):
    """Load Stable-Baselines3 PPO model and extract actor MLP weights."""
    from stable_baselines3 import PPO
    
    model = PPO.load(model_path, device="cpu")
    policy = model.policy

    # Extract MLP layers from the actor
    layers = []
    for name, module in policy.mlp_extractor.policy_net.named_modules():
        if hasattr(module, "weight"):
            layers.append({
                "name": name,
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None,
            })

    # Final action network
    for name, module in policy.action_net.named_modules():
        if hasattr(module, "weight"):
            layers.append({
                "name": f"action_{name}",
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None,
            })

    input_size = layers[0]["weight"].shape[1]
    output_size = layers[-1]["weight"].shape[0]

    return {
        "layers": layers,
        "input_size": input_size,
        "output_size": output_size,
        "activation": "tanh",   # SB3 MlpPolicy default
    }

def load_torch_policy(model_path: str):
    """Load a pure PyTorch model (must be an nn.Sequential or similar MLP)."""
    import torch
    import pickle
    import torch.nn as nn

    try:
        model = torch.load(model_path, map_location="cpu")
    except pickle.UnpicklingError:
        model = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
            )
        
    model.eval()

    layers = []
    activation =None

    for name, module in model.named_modules():
        if isinstance(module,nn.Linear) and module.weight is not None:
            layers.append({
                "name": name,
                "weight": module.weight.detach().cpu().numpy(),
                "bias": module.bias.detach().cpu().numpy() if module.bias is not None else None,
            })

        if activation is None:
            if isinstance(module, nn.Tanh):
                activation = "tanh"
            elif isinstance(module, nn.ReLU):
                activation = "relu"
            elif isinstance(module, nn.LeakyReLU):
                activation = "leaky_relu"
            elif isinstance(module, nn.ELU):
                activation = "elu"
            elif isinstance(module, nn.Sigmoid):
                activation = "sigmoid"
            elif isinstance(module, nn.Softmax):
                activation = "softmax"

    if activation is None:
        activation = "relu"  # default

        

    if not layers:
        raise ValueError("No linear layers found in the model")

    input_size = layers[0]["weight"].shape[1]
    output_size = layers[-1]["weight"].shape[0]

    return {
        "layers": layers,
        "input_size": input_size,
        "output_size": output_size,
        "activation": activation,  # change if needed
    }