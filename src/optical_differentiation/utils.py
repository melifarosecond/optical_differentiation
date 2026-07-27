def conv_macs(in_c: int, out_c: int, k: int, out_h: int, out_w: int) -> int:
    return out_h * out_w * out_c * in_c * k * k


def report_frontend_savings():
    baseline_first_block = conv_macs(in_c=1, out_c=6, k=5, out_h=28, out_w=28)
    optical_first_block = 0  

    print(f"Baseline первый блок: {baseline_first_block:,} MAC")
    print(f"Optical frontend (электроника): {optical_first_block:,} MAC")
    print(f"Экономия: {baseline_first_block:,} MAC")