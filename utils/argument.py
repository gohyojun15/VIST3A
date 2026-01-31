"""
Utility functions for parsing command line arguments.
"""

import argparse
import pathlib
import textwrap

from models.stitching_layer_builder import parse_conv_spec

# -------------------------
# Types / helpers
# -------------------------


def parse_dataset(arg: str) -> tuple[str, pathlib.Path]:
    """
    Convert a CLI token of the form NAME:ROOT into (name, root)
    and validate a few basic things.
    """
    try:
        name, root = arg.split(":", 1)
    except ValueError:  # no “:”
        raise argparse.ArgumentTypeError(
            "Dataset must be NAME:PATH, e.g. dl3dv:/data/dl3dv"
        )

    root_path = pathlib.Path(root).expanduser()
    if not root_path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {root_path}")

    return name, root_path


def make_parser(
    description: str,
    *,
    formatter: type[argparse.HelpFormatter] = argparse.ArgumentDefaultsHelpFormatter,
) -> argparse.ArgumentParser:
    """
    Create a parser with a consistent formatter that shows defaults automatically.
    """
    return argparse.ArgumentParser(
        description=description,
        formatter_class=formatter,
    )


def _dedent(s: str) -> str:
    return textwrap.dedent(s).strip("\n")


# -------------------------
# Reusable argument blocks
# -------------------------


def add_model_selection_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Model selection")
    g.add_argument(
        "--feedforward_model",
        type=str,
        default="anysplat",
        choices=["anysplat"],
        help="Feedforward model to use",
    )
    g.add_argument(
        "--video_model",
        type=str,
        default="wan",
        choices=["wan"],
        help="Video model to use",
    )
    return parser


def add_run_and_logging_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Run & logging")
    g.add_argument("--global_seed", type=int, default=23, help="Global seed")
    g.add_argument(
        "--exp_name", type=str, default="wan_anysplat_stitching", help="Experiment name"
    )

    # Correct boolean handling:
    # - exposes: --wandb_logging / --no-wandb_logging
    g.add_argument(
        "--wandb_logging",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable wandb logging",
    )
    g.add_argument(
        "--wandb_project_name",
        type=str,
        default="wan+anysplat",
        help="Wandb project name",
    )
    return parser


def add_training_loop_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Training loop")
    g.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    g.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None,
        help="Path to resume checkpoint",
    )
    g.add_argument(
        "--save_path",
        type=str,
        default="trained_checkpoint/wan_anysplat_stitching",
        help="Path to save checkpoints",
    )
    return parser


def add_optimizer_args(
    parser: argparse.ArgumentParser,
    *,
    include_warmup: bool = True,
) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Optimizer")
    g.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    g.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")

    if include_warmup:
        g.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")

    return parser


def add_common_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Shared data args used by both training and eval.
    Note: dataset is defined here as append; eval can enforce required-ness after parsing if desired,
    but we keep the original behavior for eval by defining it in the eval function instead.
    """
    g = parser.add_argument_group("Data (common)")
    g.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    g.add_argument(
        "--feedforward_resolution",
        type=int,
        default=448,
        help="Image resolution for feedforward model",
    )
    # These two resolutions can differ because AnySplat and Wan VAE use different input sizes.
    return parser


def add_training_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Training-only data arguments.
    """
    add_common_data_args(parser)

    g = parser.add_argument_group("Data (training)")
    g.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    g.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        metavar="NAME:ROOT",
        help=_dedent(
            """
            Provide one or several datasets as NAME:ROOT.
            Example:
              --dataset dl3dv:/data/dl3dv \
              --dataset other:/mnt/other_ds
            """
        ),
    )
    g.add_argument(
        "--num_frames_per_unit_scene",
        type=int,
        default=13,
        help=(
            "Many frames in a sequence will be divided into units of this number of frames, "
            "and each unit will be treated as a separate scene during training"
        ),
    )
    g.add_argument(
        "--num_images_from_unit_scene",
        type=int,
        default=13,
        help="Number of images to sample from each unit scene during training",
    )
    return parser


def add_eval_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Evaluation-only data arguments.
    Kept semantics: dataset is required=True here, as in your original code.
    """
    add_common_data_args(parser)

    g = parser.add_argument_group("Data (evaluation)")
    g.add_argument(
        "--dataset",
        type=parse_dataset,
        action="append",
        metavar="NAME:ROOT",
        required=True,
        help=_dedent(
            """
            Provide a dataset as NAME:ROOT.
            Example:
              --dataset re10k:/data/re10k
            """
        ),
    )
    g.add_argument(
        "--seq_id_map",
        type=str,
        required=True,
        help="Path to the JSON file mapping sequence names to frame IDs for evaluation",
    )
    return parser


def add_stitching_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("Stitching")
    g.add_argument(
        "--stitching_layer_location",
        type=str,
        default="enc_blocks_2",
        help="Location of the stitching layer in the feedforward model",
    )
    g.add_argument(
        "--initialization_weight_path",
        type=str,
        default=None,
        help="Path to the initialization weight for the stitching layer",
    )
    g.add_argument(
        "--stitching_layer_config",
        type=parse_conv_spec,
        default="conv3d_k5x3x3_o1024_s1x2x2_p2x1x1",
        metavar="CONV_SPEC",
        help="See --help in training script for grammar.",
    )
    # parse_conv_spec converts this string into a ConvSpec object with kernel/stride/padding.
    g.add_argument(
        "--lora_config",
        type=str,
        default="r8,a16,d0.05,f0",
        help=_dedent(
            """
            LoRA config.
            r<rank>,a<alpha>,d<dropout>,b<bias>,t<targets>,f<0/1>
            Examples:
              r4,a16,d0.05
              r8,a32,tq_proj|k_proj|v_proj
            """
        ),
    )
    return parser


# -------------------------
# Parser builders
# -------------------------


def stitching_training_argument() -> argparse.ArgumentParser:
    parser = make_parser("Stitching training argument")
    add_model_selection_args(parser)
    add_run_and_logging_args(parser)
    add_training_loop_args(parser)
    add_optimizer_args(parser, include_warmup=True)
    add_training_data_args(parser)
    add_stitching_args(parser)
    return parser


def find_layer_stitching_argument() -> argparse.ArgumentParser:
    parser = make_parser("Find layer for stitching argument")

    g = parser.add_argument_group("Feature extraction")
    g.add_argument(
        "--feature_save_path",
        type=str,
        required=True,
        help="Path to save features used for searching stitching layer",
    )
    g.add_argument(
        "--iterations_for_feature_extraction",
        type=int,
        default=100,
        help=(
            "Number of iterations for feature extraction. "
            "Total data used = batch_size × iterations_for_feature_extraction"
        ),
    )

    add_model_selection_args(parser)
    add_training_data_args(parser)
    add_stitching_args(parser)
    return parser


def stitching_nvs_evaluation_argument() -> argparse.ArgumentParser:
    parser = make_parser("Stitching NVS evaluation argument")
    add_model_selection_args(parser)
    add_stitching_args(parser)
    add_eval_data_args(parser)

    g = parser.add_argument_group("Evaluation")
    g.add_argument(
        "--checkpoint_path", type=str, help="Path to the trained stitching model"
    )
    g.add_argument(
        "--output_dir",
        type=str,
        default="nvs_evaluation_results",
        help="Path to save evaluation results",
    )
    return parser


def training_vdm_argument() -> argparse.ArgumentParser:
    """
    VDM training:
    - Uses general run/logging + training loop + optimizer, but DOES NOT include warmup_steps
    - Includes training data args
    - Enforces num_frames_per_unit_scene to be 32 by default, and validates after parsing
      (keeps this module clean and avoids argparse-internal hacks).
    """
    parser = make_parser("Training VDM argument")
    # VDM fine-tuning uses the same stitching data args but no warmup schedule.
    add_run_and_logging_args(parser)
    add_training_loop_args(parser)
    add_model_selection_args(parser)
    add_stitching_args(parser)
    add_optimizer_args(parser, include_warmup=False)  # no warmup arg here
    add_training_data_args(parser)

    g = parser.add_argument_group("VDM")
    g.add_argument(
        "--text_dataset_path", type=str, help="Path to text dataset for VDM training"
    )
    g.add_argument(
        "--checkpoint_path",
        type=str,
        help="trained_checkpoint/wan_anysplat_stitching_cloud/epoch_58",
    )
    g.add_argument(
        "--qual_coeff",
        default=0.25,
        type=float,
        help="Coefficient for quality score in reward function",
    )
    g.add_argument(
        "--mse_coeff",
        default=1.0,
        type=float,
        help="Coefficient for MSE in reward function",
    )
    g.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", type=str)
    g.add_argument(
        "--save_freq",
        default=100,
        type=int,
        help="Frequency to save model checkpoints",
    )
    g.add_argument(
        "--enable_rl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable rl",
    )

    # Enforce a fixed default; validation happens after parsing (see helper below).
    parser.set_defaults(num_frames_per_unit_scene=32)

    return parser


def inference_vist3a_argument() -> argparse.ArgumentParser:
    parser = make_parser("Inference on VIST3A argument")
    add_model_selection_args(parser)
    add_stitching_args(parser)
    add_common_data_args(parser)

    g = parser.add_argument_group("Inference")

    g.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", type=str)
    g.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the trained stitching model",
    )
    g.add_argument(
        "--transformer_lora_path",
        type=str,
        required=True,
        help="Path to the LoRA weights for the transformer",
    )
    g.add_argument(
        "--input_texts_path",
        type=str,
        required=True,
        help="Path to input texts for inference",
    )
    g.add_argument(
        "--output_dir",
        type=str,
        default="inference_vist3a_results",
        help="Path to save inference results",
    )
    g.add_argument(
        "--num_frames",
        type=int,
        default=13,
        help="Number of frames to generate for each input text",
    )
    g.add_argument(
        "--flow_shift",
        type=float,
        default=5,
        help="Flow shift value for timesteps",
    )
    g.add_argument(
        "--cfg_scale",
        type=str,
        default="7.5",
        help="Classifier-free guidance scale(s), single value or comma-separated for each stage",
    )
    return parser


# -------------------------
# Optional post-parse validators
# -------------------------


def validate_vdm_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """
    Call this in your VDM training script after parse_args() to enforce constraints.
    Example:
        parser = training_vdm_argument()
        args = parser.parse_args()
        validate_vdm_args(parser, args)
    """
    if getattr(args, "num_frames_per_unit_scene", None) != 32:
        parser.error("--num_frames_per_unit_scene must be 32 for VDM training")
