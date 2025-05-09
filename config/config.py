
import argparse
# import yaml
import os



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    #  to calculate metric
    # checkpoint_step_metric
    parser.add_argument("--checkpoint_step_metric", type=int,
        default=6000, help="checkpoint_step_metric to calculate metric")
    # exepctation of timestep---number_of_timesteps
    parser.add_argument("--number_of_timesteps", type=int,
        default=3, help="number_of_timesteps to calculate exepctation of timestep")
    parser.add_argument(
        "--gathered_ref_folder",
        type=str,
        default="/apdcephfs_cq8/share_2992679/private/zlf/ouput/diffusion_dpo/ranking/sd1_5/clean/save_ref",
        help="Path to get gathered_ref_diff at diffrent timesteps. ",
    )
    parser.add_argument("--train_batch_size_cal_metric", type=int,
        default=64, help="train_batch_size when used to calculate metrics. ")


    # comparison
    parser.add_argument("--label_smoothing", type=float,
        default=0, help="label smoothing in robust dpo or conservative dpo.")
    parser.add_argument("--use_accumulation_metric", type=bool,
    default=False, help="whether calculate metric every half epoch instead updating every batch")

    parser.add_argument("--save_npz_chunk_size", type=int,
        default=5000, help="number_of_timesteps to calculate exepctation of timestep")

    # load noisy label
    parser.add_argument("--use_npy_noisy_label", type=bool,
        default=False, help="beta in ipo.")
    parser.add_argument("--noisy_label_path", type=str,
        default="/apdcephfs_cq10/share_1275017/chengmingxu/DPO/eval_files/1n_label_0.npy", help="beta in ipo.")


    #ipo beta
    parser.add_argument("--beta", type=float,
        default=100, help="beta in ipo.")


    #sdxl
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--is_turbo",
        action="store_true",
        help=("Use if tuning SDXL Turbo instead of SDXL"),
    )


    #EMA
    parser.add_argument("--weight_type", type=str,
        default="linear", help="choose from quardratic, linear, sigmoid")



    parser.add_argument("--use_std", type=bool,
        default=False, help="Whether to use EMA model.")
    
    parser.add_argument("--use_ema", type=bool,
        default=False, help="Whether to use EMA model.")
    
    parser.add_argument("--ema_warmup_steps", type=int,
        default=2, help="Whether to use EMA model.")
    
    parser.add_argument("--use_weight_std_standarization", type=bool,
        default=False, help="Whether to use std standarization on uncertainty/weight.")
    
    parser.add_argument("--use_logits_normalization", type=bool,
        default=False, help="Whether to use std standarization on uncertainty/weight.")
    
    parser.add_argument("--weight_normalization", type=bool,
        default=False, help="Whether to use weight_normalization on uncertainty/weight.")
    
    #margin
    parser.add_argument("--use_margin", type=bool,
        default=False, help="Whether to use margin loss.")
    parser.add_argument("--margin_type", type=str,
        default= "quadratic", help="Type of margin, choose from ['linear','quadratic'].")
    
    
    #ROBUST
    parser.add_argument(
        "--use_un_reweight",
        type=bool,
        default=False,
        help=(
            "whether to use uncertainty reweight"
        ),
    )
    parser.add_argument(
        "--use_no_label_loss",
        type=bool,
        default=False,
        help=(
            "whether to use non_label loss"
        ),
    )
    parser.add_argument(
        "--flip_un_label",
        type=bool,
        default=False,
        help=(
            "whether to use uncertainty reweight"
        ),
    )
    parser.add_argument(
        "--flip_theshold",
        type=float,
        default=1,
        help=(
            "needed if flip_un_label is true"
        ),
    )
    parser.add_argument(
        "--flip_ratio_per_batch",
        type=float,
        default=1,
        help=(
            "needed if flip_un_label is true"
        ),
    )
    parser.add_argument(
        "--change_flip_weight",
        type=str,
        default=None,
        help=(
            "Choose from ['constant','inverse',None]"
        ),
    )
    parser.add_argument(
        "--flip_weight_as_constant",
        type=float,
        default=0.8,
        help=(
            "needed if change_flip_weight is 'constant'"
        ),
    )
    parser.add_argument(
        "--use_self_logit_reweight",
        type=bool,
        default=False,
        help=(
            "whether to use uncertainty reweight"
        ),
    )
    parser.add_argument(
        "--logits_rho",
        type=float,
        default=1e4,
        help=(
            "whether to use uncertainty reweight"
        ),
    )
    parser.add_argument(
        "--uncertainty_file_path",
        type=str,
        default="/apdcephfs_cq8/share_2992679/private/zlf/datasets_all_kinds/dpo_train_noisy_preprocess/2_noisy_uncertainty_result.npy",
        help=(
            "Required when use_un_reweight is TRUE. "
        ),
    )
    parser.add_argument(
        "--rewight_intercept",
        type=float,
        default=1,
        help=(
            "Required when use_un_reweight is TRUE. "
        ),
    )
    parser.add_argument(
        "--rewight_temparature",
        type=float,
        default=10,
        help=(
            "Required when use_un_reweight is TRUE. "
        ),
    )


    #sd1.5
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_split_name",
        type=str,
        default="validation",
        help="Dataset split to be used during training. Helpful to specify for conducting experimental runs.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--run_validation",
        default=True,
        action="store_true",
        help="Whether to run validation inference in between training and also after training. Helps to track progress.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=200,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--start_train_sample",
        type=int,
        default=0,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="diffusion-dpo-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--vae_encode_batch_size",
        type=int,
        default=8,
        help="Batch size to use for VAE encoding of the images for efficient processing.",
    )
    parser.add_argument(
        "--no_hflip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to random crop the input images to the resolution. If not set, the images will be center-cropped."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--beta_dpo",
        type=int,
        default=2500,
        help="DPO KL Divergence penalty.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        help="DPO loss type. Can be one of 'sigmoid' (default), 'ipo', or 'cpo'",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="diffusion-dpo-lora",
        help=("The name of the tracker to report results to."),
    )
    
    

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    args.ema_warmup_steps = args.ema_warmup_steps *args.gradient_accumulation_steps
    
    
    if args.dataset_name is None:
        raise ValueError("Must provide a `dataset_name`.")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    

    if args.use_un_reweight == True and args.uncertainty_file_path == None:
        raise ValueError("You must provide a `uncertainty_file_path` when use_un_reweight.")


    return args



