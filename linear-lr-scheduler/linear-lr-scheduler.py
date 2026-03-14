def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # Write code here
    if step == total_steps == warmup_steps:
        lr = final_lr
    elif step < warmup_steps:
        lr = initial_lr * (step / warmup_steps)
    elif warmup_steps <= step <= total_steps:
        lr = initial_lr - (initial_lr - final_lr) * (step - warmup_steps) / (total_steps - warmup_steps)
    else:
        lr = final_lr
    return lr