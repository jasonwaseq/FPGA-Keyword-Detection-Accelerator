#!/usr/bin/env python3
"""
export_weights.py - Quantize a trained checkpoint and emit the deployment
artifact set (identical file formats to model/gen_weights.py):

    weights/conv_weights.mem, dense_weights.mem   INT8 $readmemh images
    weights/conv_bias.mem,   dense_bias.mem       INT32 biases + M/S params
    weights/kws_weights.h                         C reference-model header
    weights/model_params.json                     provenance + metadata

Quantization: symmetric per-tensor INT8 for weights; biases in the INT32
accumulator domain; per-layer requantization (M, S) calibrated with the same
routine used for the bring-up weights (model/kws_quant.py), so hardware,
host reference and export can never diverge.

After exporting, rebuild everything (the bitstream bakes the ROMs and the
host binary bakes the header):  make weights?  no - make bit host sim
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
import kws_quant as q


def quant_tensor_sym(t: torch.Tensor):
    """Symmetric per-tensor INT8: returns (int_list, scale)."""
    scale = float(t.abs().max()) / 127.0
    if scale == 0.0:
        scale = 1.0
    qt = torch.clamp(torch.round(t / scale), -128, 127).to(torch.int32)
    return qt, scale


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--feat-scale", type=float, default=20.0,
                    help="host INT8 units per feature std-dev (kws.ini "
                         "quant_scale; defines the feature scale 1/x)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["state_dict"]

    conv_w_f = sd["conv.weight"]            # [8, 40, 3]
    conv_b_f = sd["conv.bias"]              # [8]
    fc_w_f = sd["fc.weight"]                # [4, 120] (time-major flatten)
    fc_b_f = sd["fc.bias"]                  # [4]

    s_feat = 1.0 / args.feat_scale

    # --- conv layer -----------------------------------------------------------
    cw_q, s_cw = quant_tensor_sym(conv_w_f)
    conv_w = [[[int(cw_q[oc, ic, k]) for ic in range(q.NUM_MFCC)]
               for k in range(q.CONV_K)] for oc in range(q.CONV_OUT_CH)]
    # bias in the accumulator domain: acc_int = acc_float / (s_feat * s_cw)
    conv_b = [int(round(float(conv_b_f[oc]) / (s_feat * s_cw)))
              for oc in range(q.CONV_OUT_CH)]

    # --- dense layer ------------------------------------------------------------
    dw_q, s_dw = quant_tensor_sym(fc_w_f)
    dense_w = [[int(dw_q[c, i]) for i in range(q.DENSE_IN)]
               for c in range(q.NUM_CLASSES)]
    # Post-conv activations are INT8 with an implicit scale set by the conv
    # requantization target; bias scaling uses a nominal activation scale of
    # 1 INT8 unit, then calibration absorbs the residual factor into (M, S).
    dense_b = [int(round(float(fc_b_f[c]) / s_dw))
               for c in range(q.NUM_CLASSES)]

    # --- requantization calibration (same procedure as bring-up weights) --------
    model = q.calibrate(conv_w, conv_b, dense_w, dense_b,
                        random.Random(20260712))
    origin = (f"export_weights.py ckpt={os.path.basename(args.ckpt)} "
              f"val_acc={ckpt.get('val_acc', 0.0):.4f} "
              f"keywords={ckpt.get('keywords')}")
    q.emit(model, args.out, origin=origin)

    print(f"exported to {args.out}/ ({origin})")
    print("rebuild: make bit host sim   # ROMs, reference model, regression")


if __name__ == "__main__":
    main()
