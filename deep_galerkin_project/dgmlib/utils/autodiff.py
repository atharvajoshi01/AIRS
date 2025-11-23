"""
Automatic differentiation utilities for computing PDE derivatives.
"""

import torch


def compute_gradient(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute first-order gradient ∂outputs/∂inputs.

    Args:
        outputs: Tensor of shape (N, 1) or (N,).
        inputs: Tensor of shape (N, 1) or (N,) with requires_grad=True.

    Returns:
        Gradient tensor of same shape as inputs.
    """
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


def compute_hessian(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute second-order derivative ∂²outputs/∂inputs².

    Args:
        outputs: Tensor of shape (N, 1) or (N,).
        inputs: Tensor of shape (N, 1) or (N,) with requires_grad=True.

    Returns:
        Hessian diagonal tensor of same shape as inputs.
    """
    # First derivative
    grad_outputs = torch.ones_like(outputs)
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative
    hess = torch.autograd.grad(
        outputs=grad,
        inputs=inputs,
        grad_outputs=torch.ones_like(grad),
        create_graph=True,
        retain_graph=True,
    )[0]

    return hess


def compute_mixed_derivative(
    outputs: torch.Tensor,
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mixed derivative ∂²outputs/∂input1∂input2.

    Args:
        outputs: Tensor of shape (N, 1) or (N,).
        input1: First input tensor with requires_grad=True.
        input2: Second input tensor with requires_grad=True.

    Returns:
        Mixed derivative tensor.
    """
    # First derivative w.r.t. input1
    grad1 = torch.autograd.grad(
        outputs=outputs,
        inputs=input1,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Derivative of grad1 w.r.t. input2
    mixed_deriv = torch.autograd.grad(
        outputs=grad1,
        inputs=input2,
        grad_outputs=torch.ones_like(grad1),
        create_graph=True,
        retain_graph=True,
    )[0]

    return mixed_deriv
