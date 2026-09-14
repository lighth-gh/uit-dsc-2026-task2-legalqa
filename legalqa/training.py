import math
import os
from pathlib import Path

from .io import copy_file, digest, file_hash, load_questions, source_hash, write_json
from .models import audit_models, load_generator, model_lock
from .prompts import pack_prompt
from .retrieval import read_retrieval
from .runtime import should_pause


def select_training_questions(questions, c):
    maximum = int(c["training"].get("max_examples",len(questions)))
    if maximum <= 0:
        raise ValueError("training.max_examples must be positive")
    ordered = sorted(questions,key=lambda key:digest({"seed":c["seed"],"id":key}))[:maximum]
    return {key:questions[key] for key in ordered}


def prepare_training_subset(c, train_path, output):
    questions = select_training_questions(load_questions(train_path,answers=True),c)
    output = Path(output)
    question_path = output.with_name(output.stem+".questions.json")
    write_json(output,questions)
    write_json(question_path,{key:{"question":item["question"]} for key,item in questions.items()})
    return {"samples":len(questions),"train":str(output),"questions":str(question_path)}


def training_examples(questions, records, tokenizer, c):
    samples, skipped, stats = [], [], []
    limit = c["training"]["max_sequence_tokens"]
    training_prompt_limit = c["training"].get("max_prompt_tokens",c["generation"]["max_input_tokens"])
    for key,item in questions.items():
        answer_ids = tokenizer(item["answer"],add_special_tokens=False)["input_ids"]+[tokenizer.eos_token_id]
        prompt_budget = min(c["generation"]["max_input_tokens"],training_prompt_limit,
                            limit-len(answer_ids))
        if prompt_budget < c["generation"]["min_context_tokens"]+256:
            skipped.append({"id":key,"reason":"full reference does not fit without losing the context", "answer_tokens":len(answer_ids)})
            continue
        prompt_ids,packed = pack_prompt(item["question"],records[key]["contexts"],tokenizer,c,prompt_budget)
        if not packed:
            skipped.append({"id":key,"reason":"no retrieved context"})
            continue
        input_ids = prompt_ids+answer_ids
        if len(input_ids)>limit:
            raise AssertionError("Training sequence exceeds configured length")
        samples.append({"input_ids":input_ids,"attention_mask":[1]*len(input_ids),
                        "labels":[-100]*len(prompt_ids)+answer_ids})
        stats.append({"id":key,"prompt_tokens":len(prompt_ids),"answer_tokens":len(answer_ids)})
    if not samples:
        raise ValueError("No complete original QA targets fit the training budget")
    return samples,{"used":len(samples),"skipped":skipped,"lengths":stats}


def training_layout(settings, environ, visible_gpus):
    world = int(environ.get("WORLD_SIZE", "1"))
    rank = int(environ.get("RANK", "0"))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    expected = int(settings.get("world_size", 2))
    if world != expected or visible_gpus < world or not 0 <= rank < world or not 0 <= local_rank < world:
        raise ValueError(f"QLoRA requires {expected} GPU workers; got world_size={world}, "
                         f"visible_gpus={visible_gpus}. Use torchrun --standalone "
                         f"--nproc_per_node={expected} --module legalqa ... fit ...")
    accumulation = int(settings["gradient_accumulation"])
    if accumulation < world or accumulation % world:
        raise ValueError("training.gradient_accumulation must be divisible by world_size to preserve effective batch")
    return world, rank, local_rank, accumulation // world


def pause_training(torch, device, world):
    stop = torch.tensor(int(should_pause()), device=device)
    if world > 1:
        torch.distributed.all_reduce(stop, op=torch.distributed.ReduceOp.MAX)
    return bool(stop.item())


def fit(c, train_path, retrieval_path, root, output, device="cuda:0", resume=None):
    if not c["generation"].get("load_in_4bit"):
        raise ValueError("QLoRA is required: generation.load_in_4bit must be true")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, TrainerCallback, set_seed
    t = c["training"]
    world, rank, local_rank, accumulation = training_layout(t, os.environ, torch.cuda.device_count())
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group(backend="nccl")
    if rank == 0:
        audit_models(c, root)
    if world > 1:
        torch.distributed.barrier()
    print(f"QLoRA rank={rank}/{world} local_rank={local_rank} device={device} "
          f"GPU={torch.cuda.get_device_name(local_rank)} accumulation={accumulation} "
          f"effective_batch={t['batch_size']*accumulation*world}", flush=True)
    set_seed(c["seed"])
    questions = select_training_questions(load_questions(train_path,answers=True),c)
    records,retrieval_id = read_retrieval(retrieval_path,questions,c,root,expected_mode=c["training"]["retrieval_mode"])
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("Training directory is nonempty. Choose a new directory or --resume an exact checkpoint.")
    # Every rank checks the empty directory before rank 0 creates reports.
    if world > 1:
        torch.distributed.barrier()
    model,tokenizer = load_generator(c,root,device,training=True)
    samples,report = training_examples(questions,records,tokenizer,c)
    output.mkdir(parents=True,exist_ok=True)
    manifest = {"train_file_sha256":file_hash(train_path),"qa_hash":digest(questions),"qa_ids":list(questions),
                "retrieval":retrieval_id,"models":model_lock(c,root),"config":c,"code":source_hash(),
                "policy":"One original BTC question/answer per example. No synthetic targets or data augmentation."}
    from .io import read_json
    if resume:
        resume = Path(resume)
        previous = read_json(resume.parent/"training_manifest.json")
        if previous!=manifest:
            raise ValueError("Resume training fingerprint differs")
    if rank == 0:
        write_json(output/"training_manifest.json",manifest)
        write_json(output/"training_data_report.json",report)
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant":False})
    model = get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=t["lora_rank"],lora_alpha=t["lora_alpha"],
                    lora_dropout=t["lora_dropout"],target_modules=t["target_modules"],bias="none"))
    model.config.use_cache=False
    adapter_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    budget = read_json(Path(root)/"parameter_audit.json")
    if adapter_parameters != budget["lora_additional"]:
        raise ValueError("Actual trainable LoRA size differs from the audited architecture")

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        values = {}
        for key,pad in [("input_ids",tokenizer.pad_token_id),("attention_mask",0),("labels",-100)]:
            values[key] = torch.tensor([row[key]+[pad]*(length-len(row[key])) for row in batch],dtype=torch.long)
        return values

    bounded = bool(os.environ.get("LEGALQA_DEADLINE"))

    class BudgetCallback(TrainerCallback):
        saved_step = -1

        def on_step_end(self, args, state, control, **kwargs):
            if pause_training(torch, device, world):
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            # An interrupted epoch also triggers this hook: save it for resume,
            # but never advertise it as a completed epoch for model selection.
            control.should_save = state.global_step != self.saved_step
            return control

        def on_save(self, args, state, control, **kwargs):
            self.saved_step = state.global_step
            if not state.is_world_process_zero:
                return control
            epoch = float(state.epoch or 0)
            if epoch >= 1 and abs(epoch-round(epoch)) < 1e-6:
                checkpoint = output/f"checkpoint-{state.global_step}"
                target = output/f"epoch-{int(round(epoch)):02d}"
                target.mkdir(exist_ok=True)
                for name in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
                    copy_file(checkpoint/name,target/name)
                write_json(target/"epoch_complete.json", {"epoch":epoch,"step":state.global_step,
                    "adapter":{name:file_hash(target/name) for name in ("adapter_config.json","adapter_model.safetensors")}})
            self.saved_step = state.global_step
            return control

    args = TrainingArguments(output_dir=str(output),num_train_epochs=t["epochs"],learning_rate=t["learning_rate"],
        per_device_train_batch_size=t["batch_size"],gradient_accumulation_steps=accumulation,
        warmup_ratio=t["warmup_ratio"],lr_scheduler_type="cosine",fp16=True,bf16=False,
        gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
        optim="paged_adamw_8bit",
        max_grad_norm=.3,save_strategy="steps" if bounded else "epoch",save_steps=10,save_total_limit=2,logging_steps=10,
        eval_strategy="no",report_to="none",remove_unused_columns=False,seed=c["seed"],data_seed=c["seed"],
        dataloader_num_workers=0,local_rank=local_rank if world > 1 else -1,
        ddp_find_unused_parameters=False,ddp_broadcast_buffers=False,
        average_tokens_across_devices=True)
    trainer = Trainer(model=model,args=args,train_dataset=samples,data_collator=collate,processing_class=tokenizer,
                      callbacks=[BudgetCallback()])
    if trainer.accelerator.num_processes != world or trainer.accelerator.device.index != local_rank:
        raise RuntimeError("Trainer did not bind every QLoRA worker to its assigned GPU")
    trainer.train(resume_from_checkpoint=resume)
    if world > 1:
        if not isinstance(trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel):
            raise RuntimeError("QLoRA completed without the required DDP model wrapper")
        # Confirm that both replicas actually completed optimizer steps.
        proof = [None] * world
        torch.distributed.all_gather_object(proof, {"rank":rank,"device":device,
            "gpu":torch.cuda.get_device_name(local_rank),"global_step":trainer.state.global_step,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(local_rank)})
        if {p['rank'] for p in proof} != set(range(world)) or len({p['global_step'] for p in proof}) != 1:
            raise RuntimeError("QLoRA workers did not complete the same optimizer steps")
    else:
        proof = [{"rank":rank,"device":device,"global_step":trainer.state.global_step}]
    if rank == 0:
        write_json(output/"distributed_training.json", {"world_size":world,"workers":proof,
            "effective_batch_size":t["batch_size"]*accumulation*world})
    if trainer.state.global_step < trainer.state.max_steps:
        if world > 1:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        return {"status":"paused", "step":trainer.state.global_step,"total_steps":trainer.state.max_steps}
    final = output/"adapter_last"
    trainer.save_model(str(final))
    if rank == 0:
        tokenizer.save_pretrained(final)
        write_json(final/"training_manifest.json",manifest)
        write_json(output/"training_result.json",{"trained_examples":len(samples),"trainable_parameters":adapter_parameters,
            "epochs":t["epochs"],"final_adapter":str(final),"world_size":world,
            "selection":"Evaluate epoch checkpoints by generated dev METEOR. adapter_last is not automatically best."})
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return {"trained_examples":len(samples),"skipped":len(report["skipped"]),"final_adapter":str(final)}
