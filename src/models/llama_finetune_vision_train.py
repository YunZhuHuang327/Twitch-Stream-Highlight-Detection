import os
import time
from contextlib import nullcontext
from datetime import datetime

import torch
import torch.distributed as dist
from accelerate.utils import is_xpu_available
from llama_cookbook.model_checkpointing import (
    save_fsdp_model_checkpoint_full,
    save_model_and_optimizer_sharded,
    save_model_checkpoint,
    save_optimizer_checkpoint,
    save_peft_checkpoint,
)
from llama_cookbook.utils.memory_utils import MemoryTrace
from llama_cookbook.utils.train_utils import (
    evaluation,
    profile,
    save_to_json,
    save_train_params,
)
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm

from src.models.llama_mapping import merge_input_ids_with_image_features


def train_mm(
    model,
    mm_projector,
    get_frames_features,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    wandb_run=None,
):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predictions

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    mm_projector = mm_projector.to(model.device)

    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            if mm_projector.finetuned:
                mm_projector.train()
            else:
                mm_projector.eval()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch + 1}",
                total=total_length,
                dynamic_ncols=True,
            )

            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if (
                        train_config.max_train_step > 0
                        and total_train_steps > train_config.max_train_step
                    ):
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print(
                                "max training steps reached, stopping training, total train steps finished: ",
                                total_train_steps - 1,
                            )
                        break
                    if "vid_id" in batch:
                        if get_frames_features is not None:
                            frames_features = [
                                get_frames_features(vid_id)
                                for vid_id in batch["vid_id"]
                            ]
                            batch["frames_features"] = torch.stack(frames_features)
                        del batch["vid_id"]

                    for key in batch:
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(
                                    torch.device(f"xpu:{local_rank}")
                                )
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to("xpu:0")
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to("cuda:0")

                    with autocast():
                        batch["inputs_embeds"] = model.model.model.embed_tokens(
                            batch["input_ids"]
                        )
                        if "frames_features" in batch:
                            if len(batch["frames_features"].shape) == 3:
                                # frames_features must be of shape (batch_size, num_frames, #tokens,,embed_dim)
                                batch["frames_features"] = batch[
                                    "frames_features"
                                ].unsqueeze(2)
                            else:
                                assert len(batch["frames_features"].shape) == 4

                            image_features = mm_projector(batch["frames_features"])
                            image_features = image_features.to(model.device).squeeze(0)

                            (
                                final_embedding,
                                final_attention_mask,
                                final_labels,
                                position_ids,
                            ) = merge_input_ids_with_image_features(
                                image_features,
                                batch["inputs_embeds"],
                                batch["input_ids"],
                                batch["attention_mask"],
                                labels=batch["labels"],
                            )
                            batch["inputs_embeds"] = final_embedding
                            batch["attention_mask"] = final_attention_mask
                            batch["labels"] = final_labels
                            # batch["position_ids"] = position_ids
                            del batch["frames_features"]

                        del batch["input_ids"]
                        loss = model(**batch).loss
                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(
                            float(torch.exp(loss.detach().float()))
                        )
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                            train_dataloader
                        ) - 1:
                            if (
                                train_config.gradient_clipping
                                and train_config.gradient_clipping_threshold > 0.0
                            ):
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(
                                        train_config.gradient_clipping_threshold
                                    )
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(),
                                        train_config.gradient_clipping_threshold,
                                    )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(
                            train_dataloader
                        ) - 1:
                            if (
                                train_config.gradient_clipping
                                and train_config.gradient_clipping_threshold > 0.0
                            ):
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(
                                        train_config.gradient_clipping_threshold
                                    )
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        model.parameters(),
                                        train_config.gradient_clipping_threshold,
                                    )
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run and (not train_config.enable_fsdp or rank == 0):
                        wandb_run.log(
                            {
                                "train/epoch": epoch + 1,
                                "train/step": epoch * len(train_dataloader) + step,
                                "train/loss": loss.detach().float(),
                            }
                        )

                    pbar.set_description(
                        f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                    )

                    if train_config.save_metrics:
                        save_to_json(
                            metrics_filename,
                            train_step_loss,
                            train_loss,
                            train_step_perplexity,
                            train_prep,
                            val_step_loss,
                            val_loss,
                            val_step_perplexity,
                            val_prep,
                        )
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if (
            is_xpu_available()
            and (torch.xpu.device_count() > 1 and train_config.enable_fsdp)
            or torch.cuda.device_count() > 1
            and train_config.enable_fsdp
        ):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(
                model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run
            )
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = (
                train_config.save_model and eval_epoch_loss < best_val_loss
            )

        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank == 0:
                        print("we are about to save the PEFT modules")
                else:
                    print("we are about to save the PEFT modules")
                save_peft_checkpoint(model, train_config.output_dir)
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(
                            f"PEFT modules are saved in {train_config.output_dir} directory"
                        )
                else:
                    print(
                        f"PEFT modules are saved in {train_config.output_dir} directory"
                    )

            else:
                if not train_config.enable_fsdp:
                    save_model_checkpoint(model, train_config.output_dir)

                elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                    print("=====================================================")
                    save_fsdp_model_checkpoint_full(
                        model, optimizer, rank, train_config, epoch=epoch
                    )

                    if train_config.save_optimizer:
                        print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )

                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    if train_config.save_optimizer:
                        print(
                            " Saving the FSDP model checkpoints using SHARDED_STATE_DICT"
                        )
                        print("=====================================================")
                        save_model_and_optimizer_sharded(
                            model, rank, train_config, optim=optimizer
                        )
                    else:
                        print(
                            " Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT"
                        )
                        print("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config)

            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch + 1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank == 0:
                print(
                    f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
                )
        else:
            print(
                f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"] = TFlops
    # saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results
