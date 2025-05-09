#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 bram-w, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import base64
import contextlib
import io
import logging
import math
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset,load_from_disk
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import collections

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    # DiffusionPipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
import sys
import ImageReward as RM 
# from utils.pickscore_utils import Selector

sys.path.append("/apdcephfs_cq8/share_2992679/private/zlf/spr_model/adaptive_dpo/")
sys.path.append("/apdcephfs_cq8/share_2992679/private/zlf/spr_model/")
print(sys.path)
# __package__="/share/home/yjq/zlf/model/spr_diffusionDPO"
# sys.path.append(__package__)
# __name__="lora_spr"
print("In module products __package__, __name__ ==", __package__,"!", __name__)


from adaptive_dpo.config.config import parse_args

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__)


VALIDATION_PROMPTS = [
    'The joker holding pistols, vray, fantasy art, art by Russ Mills, blending, smooth, serious, detailed expressions, artstyle, detailed eyes, HDR, UHD, 64k, RTX, sharp, sharp focus, highly detailed, intricate detail, professional, artistic flow, ultra detailed, high resolution illustration'
]

# validation_dataset = load_dataset("/apdcephfs_cq8/share_2992679/dataset/pickapic_v2/validation_unique",
#                                   cache_dir="/dockerdata",
#                                   split="validation")
# # VALIDATION_PROMPTS = validation_dataset["caption"][:10]





def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(args, unet, accelerator, weight_dtype, epoch, is_final_validation=False):
    logger.info(f"Running validation... \n Generating images with prompts:\n" f" {VALIDATION_PROMPTS}.")

    # create pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.safety_checker = None
    if not is_final_validation:
        pipeline.unet = accelerator.unwrap_model(unet)
    else:
        pipeline.load_lora_weights(args.output_dir, weight_name="pytorch_lora_weights.safetensors")

    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    # pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    context = contextlib.nullcontext() if is_final_validation else torch.cuda.amp.autocast()
    # torch.set_grad_enabled(False)

    image_rewards = []
    pickscores = []
    # num = int(len(VALIDATION_PROMPTS)/accelerator.num_processes)
    # if accelerator.num_processes == (accelerator.process_index + 1):
    #     VALIDATION_PROMPTS = VALIDATION_PROMPTS[accelerator.process_index*num:]
    # else:
    #     VALIDATION_PROMPTS = VALIDATION_PROMPTS[accelerator.process_index*num:(accelerator.process_index+1)*num]
    for prompt in VALIDATION_PROMPTS:
        with context:
            image = pipeline(prompt,generator=generator,guidance_scale=5).images[0]
            images.append(image)

            #计算image-reward和pickscore
            # image_reward=rm_model.score(prompt,image)
            # image_rewards.append(image_reward)
            # score = ps_selector.score(image, prompt)
            # pickscores.append(score[0])
            # print(image_rewards)
            # print(pickscores)

    


    #计算image——reward和pickscore
    tracker_key = "test" if is_final_validation else "validation"

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    tracker_key: [
                        wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}") for i, image in enumerate(images)
                    ]
                }
            )

    # Also log images without the LoRA params for comparison.
    if is_final_validation:
        pipeline.disable_lora()
        no_lora_images = [
            pipeline(prompt,generator=generator,guidance_scale=5).images[0] for prompt in VALIDATION_PROMPTS
        ]

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in no_lora_images])
                tracker.writer.add_images("test_without_lora", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "test_without_lora": [
                            wandb.Image(image, caption=f"{i}: {VALIDATION_PROMPTS[i]}")
                            for i, image in enumerate(no_lora_images)
                        ]
                    }
                )



def tokenize_captions(tokenizer, examples):
    max_length = tokenizer.model_max_length
    captions = []
    for caption in examples["caption"]:
        captions.append(caption)

    text_inputs = tokenizer(
        captions, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt"
    )

    return text_inputs.input_ids


@torch.no_grad()
def encode_prompt(text_encoder, input_ids):
    text_input_ids = input_ids.to(text_encoder.device)
    attention_mask = None

    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds


def main(args):
    # if args.run_validation:
    #     rm_model = RM.load("ImageReward-v1.0", download_root = "~/.cache/ImageReward").to("cuda")

    #     ps_selector = Selector('cpu')
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)


    # if args.use_std:
    #     std_unet1 = UNet2DConditionModel.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    #     )
    #     std_unet2 = UNet2DConditionModel.from_pretrained(
    #         args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    #     )
    #     ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)


    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Set up LoRA.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    # print(unet_lora_config)
    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)

    if args.mixed_precision == "fp16":
        for param in unet.parameters():
            # only upcast trainable parameters (LoRA) into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder atten layers
            unet_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            LoraLoaderMixin.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=None,
            )

    def load_model_hook(models, input_dir):
        unet_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(accelerator.unwrap_model(unet))):
                unet_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
        LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    if "json" in args.dataset_name:
        train_dataset = load_dataset("json", data_files=args.dataset_name,cache_dir = "/dockerdata")
        train_dataset = train_dataset["train"]
        # new_labels = np.load("/apdcephfs_cq8/share_2992679/private/chengmingxu/Datasets/DPO_10-10/w1024_h1024/不指定风格/train_label.npy").tolist()  # 这是一个示例列表，长度应与 train_dataset 的长度相同

        # # 确保新列表的长度与数据集的长度相同
        # if len(new_labels) != len(train_dataset):
        #     raise ValueError("新列表的长度必须与数据集的长度相同。")

        # # 替换 "label_0" 列
        # train_dataset = train_dataset.map(lambda x: {'label_0': new_labels})
    else:
        train_dataset = load_from_disk(
            args.dataset_name,
            # cache_dir="/dockerdata",
            # split=args.dataset_split_name,
        )
    
    # print(train_dataset)
    train_transforms = transforms.Compose(
        [
            transforms.Resize(int(args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution) if args.random_crop else transforms.CenterCrop(args.resolution),
            transforms.Lambda(lambda x: x) if args.no_hflip else transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        all_pixel_values = []
        if "json" in args.dataset_name:
            l = ["image_0", "image_1"]
        else:
            l = ["jpg_0", "jpg_1"]

        for col_name in l:
            # images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            if "json" in args.dataset_name:
                images = [Image.open(io.BytesIO(base64.b64decode(im_bytes))).convert("RGB") for im_bytes in examples[col_name]]
            else:
                images = [Image.open(io.BytesIO(im_bytes)).convert("RGB") for im_bytes in examples[col_name]]
            pixel_values = [train_transforms(image) for image in images]
            all_pixel_values.append(pixel_values)

        # Double on channel dim, jpg_y then jpg_w
        im_tup_iterator = zip(*all_pixel_values)
        combined_pixel_values = []
        for im_tup, label_0 in zip(im_tup_iterator, examples["label_0"]):
            if label_0 == 0:
                im_tup = im_tup[::-1]
            combined_im = torch.cat(im_tup, dim=0)  # no batch dim
            combined_pixel_values.append(combined_im)
        examples["pixel_values"] = combined_pixel_values

        examples["input_ids"] = tokenize_captions(tokenizer, examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        if args.use_npy_noisy_label:
            label_0_new = np.load(args.noisy_label_path).tolist()
            train_dataset = train_dataset.remove_columns("label_0")
            train_dataset = train_dataset.add_column("label_0",label_0_new)
            print(train_dataset)
        # Set the training transforms
        train_dataset = train_dataset.with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        final_dict = {"pixel_values": pixel_values}
        final_dict["input_ids"] = torch.stack([example["input_ids"] for example in examples])\
        #增加一个idx在batch中作为索引，根据这个索引可以到dataset中找到对应的example来进行修改
        # final_dict["idx"] = [example["idx"] for example in examples]
        final_dict["label_0"] = [example["label_0"] for example in examples]
        return final_dict

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # if args.use_std:
    #     std_unet1.to(accelerator.device, dtype=weight_dtype)
    #     std_unet2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            print(path)
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    # def cal_un_wy(logit, label):
    #     # train_std= np.std(logit, axis=0)
    #     train_mean = sigmoid(logit)
    #     train_lab = np.abs( train_mean-label) #* train_std
    #     # print(train_std,"std!!!!")
    #     return train_lab
    def cal_un_wy(logit, label):
        train_mean = torch.sigmoid(logit).detach()
        train_lab = torch.abs(train_mean - label)
        return train_lab
    
    # def cal_un_wy_wstd(logit, label):
    #     train_std= np.std(logit, axis=0)
    #     train_mean = np.mean(sigmoid(logit),  axis=0)
    #     train_lab = np.abs( train_mean-label) * train_std
    #     return train_lab,train_std

    def cal_un_wy_wstd(logit, label):
        train_std = logit.detach().std(dim=0)
        train_std = torch.clamp(train_std, max=1)
        train_mean = torch.sigmoid(logit).detach().mean(dim=0)
        train_lab = torch.abs(train_mean - label) * train_std
        return train_lab, train_std


    def sigmoid(x):
        # 使用NumPy的clip函数将输入值限制在[-709, 709]之间，防止overflow
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))

    unet.train()


    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet.add_adapter(unet_lora_config)

        ema_unet = EMAModel(ema_unet.parameters(), decay = 0.9999,model_cls=UNet2DConditionModel, model_config=ema_unet.config)

        ema_unet2 = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet2.add_adapter(unet_lora_config)

        ema_unet2 = EMAModel(ema_unet2.parameters(),  decay = 0.5, model_cls=UNet2DConditionModel, model_config=ema_unet2.config)
        

    if args.use_ema:
        ema_unet.to(accelerator.device,dtype=weight_dtype)
        ema_unet2.to(accelerator.device,dtype=weight_dtype)

    def cal_loss(unet,noisy_model_input,timesteps,encoder_hidden_states,target):
        model_pred_new = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample.detach()
        # Compute losses.
        model_losses = F.mse_loss(model_pred_new.float(), target.float(), reduction="none")
        model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
        model_losses_w, model_losses_l = model_losses.chunk(2)

        # For logging
        # raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
        model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)
            
        return model_diff
             
    def load_latest_checkpoint_lora(std_unet1,std_unet2):
        if args.use_std:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path1 = dirs[-1] if len(dirs) > 0 else None
            path2 = dirs[-2] if len(dirs) > 0 else None

        if path1 is None:
            accelerator.print(
                f"Checkpoint 1 does not exist. set use_std == None."
            )
            args.use_std = None
        else:
            if path2 == None:
                accelerator.print(
                f"Checkpoint 2 does not exist. only use 1 latest checkpoint."
                )
                # Set up LoRA.
                # unet_lora_config1 = LoraConfig(
                #     r=args.rank,
                #     lora_alpha=args.rank,
                #     init_lora_weights="gaussian",
                #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                # )
                # lora1 = diffusers.load_lora_weights(os.path.join(os.path.join(args.output_dir, path1),"pytorch_lora_weights.safetensors"), adapter_name="lora1")
                # Add adapter and make sure the trainable params are in float32.
                std_unet1.add_adapter(unet_lora_config)
                std_unet1.load_lora_weights(os.path.join(os.path.join(args.output_dir, path1),"pytorch_lora_weights.safetensors"))


            else:
                # lora1 = diffusers.load_lora_weights(os.path.join(os.path.join(args.output_dir, path1),"pytorch_lora_weights.safetensors"), adapter_name="lora1")
                # Add adapter and make sure the trainable params are in float32.
                std_unet1.add_adapter(unet_lora_config)
                # std_unet1.add_weighted_adapter(lora1)
                std_unet1.load_lora_weights(os.path.join(os.path.join(args.output_dir, path1),"pytorch_lora_weights.safetensors"))

                # lora2 = diffusers.load_lora_weights(os.path.join(os.path.join(args.output_dir, path1),"pytorch_lora_weights.safetensors"), adapter_name="lora2")
                # Add adapter and make sure the trainable params are in float32.
                std_unet2.add_adapter(unet_lora_config)
                # std_unet2.add_weighted_adapter(lora2)
                std_unet2.load_lora_weights(os.path.join(os.path.join(args.output_dir, path2),"pytorch_lora_weights.safetensors"))


        return std_unet1,std_unet2

    if args.use_un_reweight:
        weight = np.load(args.uncertainty_file_path)
        print(args.uncertainty_file_path,"1!!!!!!!!!!!!!!!")
    for epoch in range(first_epoch, args.num_train_epochs):
        # 每轮开始定义一个新的字典记录loss,index, label信息
        sample_stats = collections.defaultdict(list)
        average_implicit_acc = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # (batch_size, 2*channels, h, w) -> (2*batch_size, channels, h, w)
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                feed_pixel_values = torch.cat(pixel_values.chunk(2, dim=1))

                latents = []
                for i in range(0, feed_pixel_values.shape[0], args.vae_encode_batch_size):
                    latents.append(
                        vae.encode(feed_pixel_values[i : i + args.vae_encode_batch_size]).latent_dist.sample()
                    )
                latents = torch.cat(latents, dim=0)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).chunk(2)[0].repeat(2, 1, 1, 1)

                # Sample a random timestep for each image
                bsz = latents.shape[0] // 2
                
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device, dtype=torch.long
                ).repeat(2)

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_model_input = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = encode_prompt(text_encoder, batch["input_ids"]).repeat(2, 1, 1)

                # Predict the noise residual
                model_pred = unet(
                    noisy_model_input,
                    timesteps,
                    encoder_hidden_states,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Compute losses.
                model_losses = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                model_losses = model_losses.mean(dim=list(range(1, len(model_losses.shape))))
                model_losses_w, model_losses_l = model_losses.chunk(2)

                # For logging
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = model_losses_w - model_losses_l  # These are both LBS (as is t)
                 
                # Reference model predictions.
                accelerator.unwrap_model(unet).disable_adapters()
                with torch.no_grad():
                    ref_preds = unet(
                        noisy_model_input,
                        timesteps,
                        encoder_hidden_states,
                    ).sample.detach()
                    ref_loss = F.mse_loss(ref_preds.float(), target.float(), reduction="none")
                    ref_loss = ref_loss.mean(dim=list(range(1, len(ref_loss.shape))))

                    ref_losses_w, ref_losses_l = ref_loss.chunk(2)
                    ref_diff = ref_losses_w - ref_losses_l
                    raw_ref_loss = ref_loss.mean()



                # Re-enable adapters.
                accelerator.unwrap_model(unet).enable_adapters()
                # index = batch["idx"]

                # Final loss.
                logits = ref_diff - model_diff


            #     if args.use_un_reweight:
            #         # print("use_un_reweight is true!!!!!!!!!!!!!!!!!!!!!!")
            #         assert max(index) <= weight.shape[0], "index not in weight!"
            #         weight = torch.tensor(weight, requires_grad=False)
            #         batch_weight = torch.reciprocal(torch.add(torch.mul(weight, args.rewight_temparature), args.rewight_intercept)).detach()
            #         # batch_weight = torch.tensor(np.reciprocal(np.add(np.multiply(weight[index], args.rewight_temparature), args.rewight_intercept)), requires_grad=False).to(accelerator.device)
            #         if args.flip_un_label:
            #             logits[weight[index] > args.flip_theshold] *= -1
            #             # batch_weight[weight[index] > args.flip_theshold] = 1
            #         loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits)).mean()
            # else:
                if args.use_self_logit_reweight:
                    logits0 = logits.detach()
                    new_logits = (logits0*args.logits_rho)
                    if args.use_ema:
                        if step >= args.ema_warmup_steps:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                            model_diff1 = cal_loss(unet,noisy_model_input,timesteps,encoder_hidden_states,target)
                            logits1 = (ref_diff - model_diff1).detach()
                            
                            ema_unet.restore(unet.parameters())

                            
                            # weight = cal_un_wy(np.stack([new_logits,new_logits1]),np.ones_like(new_logits))
                            # print(np.stack([new_logits,new_logits1]))
                                # Switch back to the original UNet parameters.
                            


                            ema_unet2.store(unet.parameters())
                            ema_unet2.copy_to(unet.parameters())
                            model_diff2 = cal_loss(unet,noisy_model_input,timesteps,encoder_hidden_states,target)
                            # print(model_diff2.requires_grad)---False
                            logits2 = (ref_diff - model_diff2).detach()
                            ema_unet2.restore(unet.parameters())
                            if args.use_logits_normalization:
                                logits_normalization = torch.stack([logits0, logits1, logits2])
                                norm = torch.norm(logits_normalization, dim=1, keepdim=True)
                                logits_normalization = logits_normalization / norm
                                weight,train_std = cal_un_wy_wstd(logits_normalization*args.logits_rho,torch.ones_like(new_logits))

                            else:
                                new_logits1 = (logits1*args.logits_rho)
                                new_logits2 = (logits2*args.logits_rho)
                                weight,train_std = cal_un_wy_wstd(torch.stack([new_logits,new_logits1,new_logits2]),torch.ones_like(new_logits))
                            if args.use_weight_std_standarization:
                                std_mean = torch.mean(train_std)
                                if std_mean!=0:
                                    weight = weight / std_mean
                                else:
                                    weight = cal_un_wy(new_logits,torch.ones_like(new_logits))
                            if args.weight_normalization:
                                sum_weight = torch.sum(weight)  # 计算权重的和
                                weight = weight / sum_weight   
                        else:
                            weight = torch.zeros_like(new_logits)
                            train_std = torch.zeros_like(new_logits)
                            logits1 = torch.zeros_like(new_logits)
                            logits2 = torch.zeros_like(new_logits)
                    else:
                        weight = cal_un_wy(new_logits,torch.ones_like(new_logits))
                    if args.flip_un_label:
                        if (args.use_ema and step >= args.ema_warmup_steps)  or (not args.use_ema):
                            threshold_1,indices = torch.sort(weight)
                            threshold_1 = threshold_1[int(weight.shape[0] * (1 - args.flip_ratio_per_batch))]
                            threshold = max(args.flip_theshold,threshold_1.item())
                            indices = torch.where(weight > threshold)[0]
                            logits[indices] *= -1 
                            if args.change_flip_weight != None:
                                if args.change_flip_weight == "constant":
                                    
                                    weight[weight > threshold] = args.flip_weight_as_constant
                                elif args.change_flip_weight == "inverse":
                                    # print("inverse!!!!!!")
                                    weight[weight > threshold] *= -1
                                    weight[weight <- threshold] += 1
                    if args.weight_type == "linear":
                        batch_weight = torch.reciprocal(torch.add(torch.mul(weight, args.rewight_temparature), args.rewight_intercept)).detach()
                    elif args.weight_type == "quardratic":
                        batch_weight = torch.reciprocal(torch.add(torch.mul(weight*weight, args.rewight_temparature), args.rewight_intercept)).detach()
                    elif args.weight_type == "sqrt":
                        batch_weight = torch.reciprocal(
                            torch.add(
                                torch.exp(args.rewight_temparature * torch.sqrt(weight)),  # 使用 sqrt 替换平方项
                                args.rewight_intercept
                            )
                        ).detach()
                        
                    elif args.weight_type == "sigmoid":
                        # batch_weight = torch.reciprocal(torch.add(torch.mul(weight*weight, args.rewight_temparature), args.rewight_intercept)).detach()
                        batch_weight = torch.reciprocal(
                            torch.add(
                                torch.exp(args.rewight_temparature * weight),  # 使用 exp 替换平方项
                                args.rewight_intercept
                            )
                        ).detach()
                    else:
                            # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits) ).mean()
                            raise ValueError("--margin_type must be chosen from ['linear', 'quadratic']")


                if args.loss_type == "sigmoid":
                    if args.use_self_logit_reweight:
                        if args.use_margin:
                            # print(logits.mean(),"mean!!!!!")
                            weight = torch.tensor(weight, requires_grad=False).to(accelerator.device)
                            if args.margin_type == 'linear':
                                loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits - (-weight/args.beta_dpo + logits.detach().mean())) ).mean()
                            elif args.margin_type == 'quadratic':
                                loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits - (-weight*weight/args.beta_dpo + logits.detach().mean())) ).mean()
                            else:
                                # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits) ).mean()
                                raise ValueError("--margin_type must be chosen from ['linear', 'quadratic']")
                                # print("margin_type must be chosen from ['linear','quadratic']")
                        else:
                            loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits) ).mean()
                        # elif (not args.use_un_reweight) and args.use_no_label_loss:
                        #     # print("no_label!!!")
                        #     #按照logits正负来翻转
                        #     # print(logits==0)
                        #     logits[logits < 0] *= -1
                        #     # print(logits==0)
                        #     logits_reverse = -logits

                        #     new_logits = (logits*args.logits_rho).cpu().detach().numpy()
                        #     # print(new_logits)
                        #     weight = cal_un_wy(new_logits,torch.ones_like(new_logits))
                        #     weight_reverse = - weight + torch.ones_like(weight)
                        #     print(weight,weight_reverse)
                        #     batch_weight = torch.reciprocal(torch.add(torch.mul(weight, args.rewight_temparature), args.rewight_intercept)).detach()
                        #     # batch_weight = torch.tensor(np.reciprocal(np.add(np.multiply(weight, args.rewight_temparature), args.rewight_intercept)), requires_grad=False).to(accelerator.device)
                        #     # batch_weight_reverse = torch.tensor(np.reciprocal(np.add(np.multiply(weight_reverse, args.rewight_temparature), args.rewight_intercept)), requires_grad=False).to(accelerator.device)
                        #     batch_weight_reverse = torch.reciprocal(torch.add(torch.mul(weight_reverse, args.rewight_temparature), args.rewight_intercept)).detach()
                        #     # batch_weight[weight > 0.8] = 0
                        #     # batch_weight_reverse[weight > 0.8] = 1

                        #     # batch_weight[weight_reverse > 0.8] = 1
                        #     # batch_weight_reverse[weight_reverse > 0.8] = 0

                        #     loss = -1 *(0.8*batch_weight* F.logsigmoid(args.beta_dpo * logits)+0.2*batch_weight_reverse *F.logsigmoid(args.beta_dpo * logits_reverse) ).mean()
                        #     print(loss)
                    else:
                        loss = -1 * F.logsigmoid(args.beta_dpo * logits).mean()
                elif args.loss_type == "hinge":
                    loss = torch.relu(1 - args.beta_dpo * logits).mean()
                elif args.loss_type == "ipo":
                    # print("ipo!!!")
                    if args.use_self_logit_reweight:
                        if args.use_margin:
                            weight = torch.tensor(weight, requires_grad=False).to(accelerator.device)
                            if args.margin_type == 'linear':
                                    losses = batch_weight*(logits + (-weight/(args.beta_dpo**2) + (logits/args.beta_dpo).detach().mean()) - 1 / (2 * args.beta)) ** 2
                                    # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits - (-weight/args.beta_dpo + logits.detach().mean())) ).mean()
                            elif args.margin_type == 'quadratic':
                                    losses = batch_weight*(logits + (-weight*weight/(args.beta_dpo**2) + (logits/args.beta_dpo).detach().mean()) - 1 / (2 * args.beta)) ** 2
                                    # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits - (-weight*weight/args.beta_dpo + logits.detach().mean())) ).mean()
                            else:
                                    # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits) ).mean()
                                    raise ValueError("--margin_type must be chosen from ['linear', 'quadratic']")
                                    # print("margin_type must be chosen from ['linear','quadratic']")
                        else:
                            losses = batch_weight*(logits - 1 / (2 * args.beta)) ** 2
                            # loss = -1 *(batch_weight* F.logsigmoid(args.beta_dpo * logits) ).mean()
                        # print(losses)
                    else:
                        losses = (logits - 1 / (2 * args.beta)) ** 2
                    loss = losses.mean()
                elif args.loss_type == "robust":
                    losses = (-1 * F.logsigmoid(args.beta_dpo * logits) * (1-args.label_smoothing) + 1 * F.logsigmoid(-args.beta_dpo * logits) * args.label_smoothing) / (1-2*args.label_smoothing)
                    loss = losses.mean()
                elif args.loss_type == "conservative":
                    losses = -1 * F.logsigmoid(args.beta_dpo * logits) * (1-args.label_smoothing) - 1 * F.logsigmoid(-args.beta_dpo * logits) * args.label_smoothing
                    loss = losses.mean()
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                implicit_acc = (logits > 0).sum().float() / logits.size(0)
                implicit_acc += 0.5 * (logits == 0).sum().float() / logits.size(0)

                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # 算两个指标保存下来分析一下
                reward_w = (ref_losses_w - model_losses_w)
                reward_l = (ref_losses_l - model_losses_l)
                # label_0 = batch["label_0"]
                
                if args.use_self_logit_reweight:
                    sample_stats["logit_w"]+= reward_w.cpu().detach().tolist()
                    sample_stats["logit_l"]+= reward_l.cpu().detach().tolist()
                    sample_stats["logits"]+= logits.cpu().detach().tolist()
                    sample_stats["uncertainty"]+= weight.cpu().detach().tolist()
                    sample_stats["batch_weight"]+= batch_weight.cpu().detach().tolist()

                    # sample_stats["index"]+= index
                    if args.use_ema:
                        sample_stats["logits1"]+= logits1.cpu().detach().tolist()
                        sample_stats["logits2"]+= logits2.cpu().detach().tolist()
                        sample_stats["train_std"]+= train_std.cpu().detach().tolist()

                # sample_stats["label_0"]+= label_0

                average_implicit_acc.append(implicit_acc.cpu().detach().item())
                average_implicit_acc_scalar = sum(average_implicit_acc)/len(average_implicit_acc)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                    ema_unet2.step(unet.parameters())

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if args.run_validation and global_step % args.validation_steps == 0:
                    log_validation(
                        args, unet=unet, accelerator=accelerator, weight_dtype=weight_dtype, epoch=epoch
                    )
                accelerator.wait_for_everyone()

            logs = {
                "loss": loss.detach().item(),
                "raw_model_loss": raw_model_loss.detach().item(),
                "ref_loss": raw_ref_loss.detach().item(),
                "implicit_acc": implicit_acc.detach().item(),
                "average_implicit_acc": average_implicit_acc_scalar,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        
            
        if args.use_self_logit_reweight:
            # 保存并且清空sample_stats
            np.save(os.path.join(args.output_dir,f"logits_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logits"]))
            np.save(os.path.join(args.output_dir,f"logit_w_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logit_w"]))
            np.save(os.path.join(args.output_dir,f"logit_l_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logit_l"]))
            # np.save(os.path.join(args.output_dir,f"logits_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logits"]))
            np.save(os.path.join(args.output_dir,f"uncertainty_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["uncertainty"]))
            np.save(os.path.join(args.output_dir,f"batch_weight_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["batch_weight"]))
            # np.save(os.path.join(args.output_dir,f"index_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["index"]))
            if args.use_ema:
                np.save(os.path.join(args.output_dir,f"logits1_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logits1"]))
                np.save(os.path.join(args.output_dir,f"logits2_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["logits2"]))
                np.save(os.path.join(args.output_dir,f"train_std_step_{global_step}_processs_{accelerator.process_index}.npy"),np.array(sample_stats["train_std"]))
#  sample_stats["uncertainty"]+= weight.cpu().detach().tolist()
                # sample_stats["batch_weight"]+= batch_weight.cpu().detach().tolist()
                # sample_stats["logits1"]+= logits1.cpu().detach().tolist()
                # sample_stats["logits2"]+= logits2.cpu().detach().tolist()
                # sample_stats["train_std"]+= train_std.cpu().detach().tolist()

        
        

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet = unet.to(torch.float32)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir, unet_lora_layers=unet_lora_state_dict, text_encoder_lora_layers=None
        )

        # Final validation?
    if args.run_validation:
        log_validation(
            args,
            unet=None,
            accelerator=accelerator,
            weight_dtype=weight_dtype,
            epoch=epoch,
            is_final_validation=True,
        )
    accelerator.wait_for_everyone()

        # if args.push_to_hub:
        #     upload_folder(
        #         repo_id=repo_id,
        #         folder_path=args.output_dir,
        #         commit_message="End of training",
        #         ignore_patterns=["step_*", "epoch_*"],
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
