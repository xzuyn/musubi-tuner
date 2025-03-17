import argparse
import os

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from utils import model_utils

import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def convert_from_diffusion_pipe(prefix, weights_sd):
    # convert from diffusion-pipe(?) to default LoRA
    # diffusion-pipe format: {"diffusion_model.module.name.lora_A.weight": weight, "diffusion_model.module.name.lora_B.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    # note: diffusion-pipe has no alpha, so alpha is set to rank
    new_weights_sd = {}
    lora_dims = {}
    for key, weight in weights_sd.items():
        diffusion_pipe_prefix, key_body = key.split(".", 1)
        if diffusion_pipe_prefix != "diffusion_model" and diffusion_pipe_prefix != "transformer":
            logger.warning(f"unexpected key: {key} in diffusion-pipe format")
            continue

        new_key = f"{prefix}{key_body}".replace(".", "_").replace("_lora_A_", ".lora_down.").replace("_lora_B_", ".lora_up.")
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]

    # add alpha with rank
    weight_dtype = weight.dtype  # use last weight dtype because no alpha
    for lora_name, dim in lora_dims.items():
        new_weights_sd[f"{lora_name}.alpha"] = torch.tensor(dim, dtype=weight_dtype)

    return new_weights_sd


def convert_from_omi(prefix, omi_prefix, weights_sd):
    # convert from OMI to default LoRA
    # OMI format: {"prefix.module.name.lora_down.weight": weight, "prefix.module.name.lora_up.weight": weight, ...}
    # default LoRA format: {"prefix_module_name.lora_down.weight": weight, "prefix_module_name.lora_up.weight": weight, ...}

    new_weights_sd = {}
    lora_dims = {}
    weight_dtype = None
    for key, weight in weights_sd.items():
        omi_prefix, key_body = key.split(".", 1)
        if omi_prefix != "diffusion":
            logger.warning(f"unexpected key: {key} in OMI format")  # T5, CLIP, etc.
            continue

        # only supports lora_down, lora_up and alpha
        new_key = (
            f"{prefix}{key_body}".replace(".", "_")
            .replace("_lora_down_", ".lora_down.")
            .replace("_lora_up_", ".lora_up.")
            .replace("_alpha", ".alpha")
        )
        new_weights_sd[new_key] = weight

        lora_name = new_key.split(".")[0]  # before first dot
        if lora_name not in lora_dims and "lora_down" in new_key:
            lora_dims[lora_name] = weight.shape[0]
            if weight_dtype is None:
                weight_dtype = weight.dtype  # use first weight dtype for lora_down

    # add alpha with rank
    for lora_name, dim in lora_dims.items():
        alpha_key = f"{lora_name}.alpha"
        if alpha_key not in new_weights_sd:
            new_weights_sd[alpha_key] = torch.tensor(dim, dtype=weight_dtype)

    return new_weights_sd


def lora_name_to_module_name(prefix, lora_name):
    module_name = lora_name[len(prefix) :]  # remove "lora_unet_"

    module_name = module_name.replace("_", ".")  # replace "_" with "."
    if ".cross.attn." in module_name or ".self.attn." in module_name:
        # Wan2.1 lora name to module name: ugly but works
        module_name = module_name.replace("cross.attn", "cross_attn")  # fix cross attn
        module_name = module_name.replace("self.attn", "self_attn")  # fix self attn
        module_name = module_name.replace("k.img", "k_img")  # fix k img
        module_name = module_name.replace("v.img", "v_img")  # fix v img
    else:
        # HunyuanVideo lora name to module name: ugly but works
        module_name = module_name.replace("double.blocks.", "double_blocks.")  # fix double blocks
        module_name = module_name.replace("single.blocks.", "single_blocks.")  # fix single blocks
        module_name = module_name.replace("img.", "img_")  # fix img
        module_name = module_name.replace("txt.", "txt_")  # fix txt
        module_name = module_name.replace("attn.", "attn_")  # fix attn
    return module_name


def convert_to_diffusion_pipe(prefix, weights_sd):
    # convert from default LoRA to diffusion-pipe

    # get alphas
    lora_alphas = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            lora_name = key.split(".", 1)[0]  # before first dot
            if lora_name not in lora_alphas and "alpha" in key:
                lora_alphas[lora_name] = weight

    new_weights_sd = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            if "alpha" in key:
                continue

            lora_name = key.split(".", 1)[0]  # before first dot

            module_name = lora_name_to_module_name(prefix, lora_name)

            diffusion_pipe_prefix = "diffusion_model"
            if "lora_down" in key:
                new_key = f"{diffusion_pipe_prefix}.{module_name}.lora_A.weight"
                dim = weight.shape[0]
            elif "lora_up" in key:
                new_key = f"{diffusion_pipe_prefix}.{module_name}.lora_B.weight"
                dim = weight.shape[1]
            else:
                logger.warning(f"unexpected key: {key} in default LoRA format")
                continue

            # scale weight by alpha
            if lora_name in lora_alphas:
                # we scale both down and up, so scale is sqrt
                scale = lora_alphas[lora_name] / dim
                scale = scale.sqrt()
                weight = weight * scale
            else:
                logger.warning(f"missing alpha for {lora_name}")

            new_weights_sd[new_key] = weight

    return new_weights_sd


def convert_to_omi(prefix, omi_prefix, weights_sd):
    # convert from default LoRA to OMI

    new_weights_sd = {}
    for key, weight in weights_sd.items():
        if key.startswith(prefix):
            # `.` separates lora_name and lora_weight_name, like "lora_unet_double_blocks_attn_q.lora_up.weight"
            lora_name, lora_weight_name = key.split(".", 1)

            module_name = lora_name_to_module_name(prefix, lora_name)

            new_key = f"{omi_prefix}.{module_name}.{lora_weight_name}"
            new_weights_sd[new_key] = weight

    return new_weights_sd


def convert(input_file, output_file, target_format):
    logger.info(f"loading {input_file}")
    weights_sd = load_file(input_file)
    with safe_open(input_file, framework="pt") as f:
        metadata = f.metadata()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    default_prefix = "lora_unet_"

    # determine original format
    original_format = None
    for key in weights_sd.keys():
        if key.startswith(default_prefix):  # default
            original_format = "default"
            break
        elif "lora_A" in key or "lora_B" in key:  # diffusion-pipe
            original_format = "other"
            break
        elif "lora_down" in key or "lora_up" in key:  # no prefix but lora_down or lora_up
            original_format = "omi"
            break

    if original_format is None:
        raise ValueError("unknown original format")

    logger.info(f"original format: {original_format}")

    if original_format == target_format:
        logger.info("nothing to do")
        return

    # determine architecture: currently supports HunyuanVideo and Wan2.1
    is_hunyuan = None
    for key in weights_sd.keys():
        if "double_blocks" in key or "single_blocks" in key:
            is_hunyuan = True
            break
        elif "cross_attn" in key or "self_attn" in key:
            is_hunyuan = False
            break

    if is_hunyuan is None:
        raise ValueError("unknown architecture")

    omi_prefix = "transformers" if is_hunyuan else "diffusion"

    # if original format isn't default, convert to default first
    if original_format != "default":
        logger.info("converting to default")
        if original_format == "other":
            new_weights_sd = convert_from_diffusion_pipe(default_prefix, weights_sd)
        elif original_format == "omi":
            new_weights_sd = convert_from_omi(default_prefix, omi_prefix, weights_sd)
        weights_sd = new_weights_sd

    # convert to target format
    logger.info(f"converting to {target_format}")
    if target_format == "default":
        new_weights_sd = weights_sd
    elif target_format == "other":
        new_weights_sd = convert_to_diffusion_pipe(default_prefix, weights_sd)
    elif target_format == "omi":
        new_weights_sd = convert_to_omi(default_prefix, omi_prefix, weights_sd)
    else:
        raise ValueError(f"unknown target format: {target_format}")

    # set-up metadata
    metadata = metadata or {}
    if "sshs_model_hash" in metadata:
        model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(new_weights_sd, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"saving to {output_file}")
    save_file(new_weights_sd, output_file, metadata=metadata)

    logger.info("done")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert LoRA weights between default and other formats")
    parser.add_argument("--input", type=str, required=True, help="input model file")
    parser.add_argument("--output", type=str, required=True, help="output model file")
    parser.add_argument("--target", type=str, required=True, choices=["other", "default", "omi"], help="target format")
    parser.add_argument("--test", action="store_true", help="run tests")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.test:
        # temporary test
        def run_tests(input_file, output_file):
            # input should be default and alpha=1

            # convert to diffusion-pipe
            diffusion_pipe_file = os.path.splitext(output_file)[0] + "_diffusion_pipe.safetensors"
            convert(input_file, diffusion_pipe_file, "other")

            # convert back to default
            dp_default_file = os.path.splitext(output_file)[0] + "_dp_default.safetensors"
            convert(diffusion_pipe_file, dp_default_file, "default")

            # convert to OMI
            omi_file = os.path.splitext(output_file)[0] + "_omi.safetensors"
            convert(input_file, omi_file, "omi")

            # convert back to default
            omi_default_file = os.path.splitext(output_file)[0] + "_omi_default.safetensors"
            convert(omi_file, omi_default_file, "default")

            # compare weights
            weights_sd = load_file(input_file)
            dp_default_weights_sd = load_file(dp_default_file)
            omi_default_weights_sd = load_file(omi_default_file)

            # scale-back by alpha for diffusion-pipe -> default weights
            alphas = {key: weight for key, weight in dp_default_weights_sd.items() if key.endswith(".alpha")}  # = rank
            for key, weight in dp_default_weights_sd.items():
                if key.endswith(".alpha"):
                    continue
                lora_name = key.split(".")[0]  # before first dot
                scale = 1.0 / alphas[lora_name + ".alpha"]
                scale = scale.sqrt()  # scale both down and up
                dp_default_weights_sd[key] = weight / scale

            for key, weight in weights_sd.items():
                dp_default_weight = dp_default_weights_sd[key]
                omi_default_weight = omi_default_weights_sd[key]

                # we can't compare directly because of scaling by alpha
                if ".alpha" not in key:
                    assert torch.allclose(
                        weight, dp_default_weight, atol=1e-3, rtol=1e-3
                    ), f"diffusion-pipe to default conversion failed for key: {key}"

                # we can compare directry for OMI
                assert (weight == omi_default_weight).all(), f"OMI to default conversion failed for key: {key}"

            # compare metadata
            with safe_open(input_file, framework="pt") as f:
                input_metadata = f.metadata()
            with safe_open(diffusion_pipe_file, framework="pt") as f:
                diffusion_pipe_metadata = f.metadata()
            with safe_open(dp_default_file, framework="pt") as f:
                dp_default_metadata = f.metadata()
            with safe_open(omi_file, framework="pt") as f:
                omi_metadata = f.metadata()
            with safe_open(omi_default_file, framework="pt") as f:
                omi_default_metadata = f.metadata()


            # hashes are changed, so remove them
            del input_metadata["sshs_model_hash"]
            del input_metadata["sshs_legacy_hash"]
            del diffusion_pipe_metadata["sshs_model_hash"]
            del diffusion_pipe_metadata["sshs_legacy_hash"]
            del dp_default_metadata["sshs_model_hash"]
            del dp_default_metadata["sshs_legacy_hash"]
            del omi_metadata["sshs_model_hash"]
            del omi_metadata["sshs_legacy_hash"]
            del omi_default_metadata["sshs_model_hash"]
            del omi_default_metadata["sshs_legacy_hash"]

            assert input_metadata == omi_metadata, "metadata mismatch between input and OMI"
            assert input_metadata == omi_default_metadata, "metadata mismatch between input and OMI to default"
            assert input_metadata == diffusion_pipe_metadata, "metadata mismatch between input and diffusion-pipe"
            assert input_metadata == dp_default_metadata, "metadata mismatch between input and diffusion-pipe to default"

            print("tests passed")

        run_tests(args.input, args.output)
    else:
        convert(args.input, args.output, args.target)
