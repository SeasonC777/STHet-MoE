"""
Quick smoke test to verify that the model can be instantiated and run.
"""
import torch
from models.himoe import HiMoE


def main():
    num_nodes = 16
    in_dim = 3
    out_dim = 1
    hidden_dim = 16
    num_experts = 4
    input_len = 12
    output_len = 3
    batch_size = 2

    static_adj = torch.eye(num_nodes)

    print("Creating HiMoE model...")
    model = HiMoE(
        num_nodes=num_nodes,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        static_adj=static_adj,
        num_experts=num_experts,
        num_layers=2,
        input_len=input_len,
        output_len=output_len,
        kernel_size=(3, 5, 7),
        temporal_dilations=(1, 2, 3),
        dropout=0.3,
        temperature=1.0,
        out_dim=out_dim,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    print("Testing forward pass...")
    x = torch.randn(batch_size, input_len, num_nodes, in_dim)
    y = model(x)

    expected_shape = (batch_size, output_len, num_nodes, out_dim)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: {expected_shape}")

    assert y.shape == expected_shape, "Output shape mismatch."
    print("\n[PASS] Model test passed.")


if __name__ == '__main__':
    main()
