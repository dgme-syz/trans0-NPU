# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from vllm import LLM, SamplingParams
from collections import OrderedDict
from configs.prompts import TRANS_PROMPTS, LABEL_MARK, make_mt_instruction
from configs.lang_codes import LangCodes
from utils.common_utils import print_once, set_special_tokens, get_path
from modules.data import read_rawline_data
from modules.agent import TransAgent
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.data import test_data_collector
from utils.infer_utils import process_flores_test, process_mix_flores_test, extract_test
import numpy as np
import torch.distributed as dist
import torch, time, json
import gc, os, glob
import pandas as pd

# -------------------- 统一设备管理 --------------------
if torch.npu.is_available():
    device = torch.device("npu:0")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# -------------------- 工具函数 --------------------
def empty_cache():
    if torch.npu.is_available():
        import torch_npu
        torch.npu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_available_memory(device_index: int):
    """返回设备可用内存 GB"""
    if torch.cuda.is_available():
        dev = torch.device(f"cuda:{device_index}")
        props = torch.cuda.get_device_properties(dev)
        used = torch.cuda.memory_allocated(dev)
        return (props.total_memory - used) / 1024.0 / 1024.0 / 1024.0
    elif torch.npu.is_available():
        import torch_npu
        dev = f"npu:{device_index}"
        props = torch_npu.npu.get_device_properties(dev)
        used = torch_npu.npu.memory_allocated(dev)
        return (props.total_memory - used) / 1024.0 / 1024.0 / 1024.0
    else:
        return 0.0

def get_device_count():
    if torch.npu.is_available():
        import torch_npu
        return torch.npu.device_count()
    elif torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1

# -------------------- LLM --------------------
lang_codes = LangCodes()

def prepare_vllm_inference(args, model_dir=None, override_cache=True, cache_suffix=""):
    target_model_path = get_path(args, args.output_dir) if model_dir is None else model_dir
    print_once(f">>> loading from >>>:{target_model_path}" )
    
    available_mem = check_available_memory(device_index=0)
    gpu_utilization = min(35.0 / max(available_mem, 1e-6), 0.9)
    n_devices = get_device_count()

    if args.use_lora:
        cache_file = os.path.join(get_path(args, args.cache_dir), "cache_merged_llm"+cache_suffix)
        if override_cache or not os.path.exists(cache_file):
            base_model = AutoModelForCausalLM.from_pretrained(
                args.llm_path, trust_remote_code=True, device_map="auto"
            )
            peft_model = PeftModel.from_pretrained(base_model, target_model_path)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(os.path.join(args.cache_dir, "cache_merged_llm"))
            merged_model.to("cpu")
            del merged_model
            gc.collect()
            empty_cache()
            print_once("release device memory")
        llm = LLM(
            model=os.path.join(get_path(args, args.cache_dir), "cache_merged_llm"),
            dtype=torch.bfloat16 if args.bf16 else torch.float16,
            tokenizer=target_model_path,
            tensor_parallel_size=n_devices,
            gpu_memory_utilization=gpu_utilization
        )
    else:
        print("preparing vllm with ", n_devices)
        llm = LLM(
            model=target_model_path,
            dtype=torch.bfloat16 if args.bf16 else torch.float16,
            tokenizer=target_model_path,
            tensor_parallel_size=n_devices,
            gpu_memory_utilization=gpu_utilization
        )
        print("done")
    return llm

# ----------------------------------------
def vllm_inference_onair(args, override_cache=False):
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    sampling_params = SamplingParams(n=1, temperature=0., max_tokens=args.max_new_tokens)
    llm = prepare_vllm_inference(args, override_cache=override_cache)
    with torch.no_grad():
        while True:
            print("please input the query:")
            input_l = input()
            tokenizer = llm.get_tokenizer()
            if tokenizer.chat_template is not None:
                input_l = tokenizer.apply_chat_template(
                    make_mt_instruction(input_l, llm_path=args.output_dir),
                    tokenize=False, add_generation_prompt=True
                )
            generation_out = llm.generate([input_l], sampling_params)
            for item in generation_out:
                for item_out in item.outputs:
                    l = item_out.text
                    print(">>> model input: >>>", input_l)
                    if LABEL_MARK in l:
                        mark_index = l.index(LABEL_MARK)
                        print(l.strip()[mark_index:].replace(LABEL_MARK, ""))
                    else:
                        print(">>>> " + l + "\n")

# ----------------------------------------
def vllm_inference(args, inputs_list, src_lang_code, trg_lang_code, override_cache=False):
    trans_prompt = TRANS_PROMPTS[0]
    input_ls = [
        trans_prompt.format(
            src_lan=lang_codes.get_lang(src_lang_code),
            trg_lan=lang_codes.get_lang(trg_lang_code),
            src_sent=l)
        for l in inputs_list
    ]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    sampling_params = SamplingParams(n=1, temperature=0, max_tokens=args.max_new_tokens)
    llm = prepare_vllm_inference(args, model_dir=None, override_cache=override_cache)
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is not None:
        input_ls = [
            tokenizer.apply_chat_template(
                make_mt_instruction(l, llm_path=args.output_dir),
                tokenize=False, add_generation_prompt=True
            ) for l in input_ls
        ]
    else:
        input_ls = [l + LABEL_MARK for l in input_ls]
    generation_out = llm.generate(input_ls, sampling_params=sampling_params)
    return generation_out

# ----------------------------------------
def distributed_inference(args, llm_dir, input_lists, src_lang_code, trg_lang_code, override_cache=False, cache_suffix=""):
    cache_path = os.path.join(get_path(args, args.cache_dir), "cached_inference"+("_"+cache_suffix if cache_suffix else ""))
    if override_cache and dist.get_rank() == 0:
        os.system(f"rm -rf {cache_path}")
    os.makedirs(cache_path, exist_ok=True)

    target_model_path = get_path(args, args.output_dir) if llm_dir is None else llm_dir
    print_once(f">>> validate trg output_dir >>>:{target_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        model_max_length=args.max_length,
        padding_side="left",
        truncation_size="left",
        trust_remote_code=True
    )

    time.sleep(int(os.environ.get("ARNOLD_WORKER_NUM", 1)) * 10)
    llm = AutoModelForCausalLM.from_pretrained(
        target_model_path, trust_remote_code=True, use_cache=True
    ).to(device)

    llm, tokenizer = set_special_tokens(llm, tokenizer)
    llm.eval()

    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(input_lists, shuffle=False)
    data_loader = DataLoader(input_lists, shuffle=False, batch_size=args.per_device_eval_batch_size, sampler=sampler)

    dist_outs = []
    progress_bar = tqdm(range(len(data_loader)), disable=(dist.get_rank() != 0))
    for _, batch_lines in enumerate(data_loader):
        progress_bar.update(1)
        processed_batch = test_data_collector(batch_lines, tokenizer=tokenizer, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code)
        input_ids = processed_batch["input_ids"].to(device)
        with torch.no_grad():
            generation_out = llm.generate(
                input_ids=input_ids,
                attention_mask=processed_batch["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True
            )
        output_seq = generation_out.sequences.reshape(input_ids.shape[0], generation_config.num_return_sequences, -1)
        input_length = input_ids.shape[1]
        output_seq = output_seq[:, :, input_length:]
        for out_l in output_seq:
            processed_out = tokenizer.batch_decode(out_l, skip_special_tokens=True)[0].replace("\n", " ")
            dist_outs.append(processed_out)

    # crazy, code lost
    with open(os.path.join(cache_path, f"rank_{dist.get_rank()}"), "w") as cache_file:
        for l in dist_outs:
            cache_file.write(l + "\n")


    empty_cache()
    dist.barrier()

    merged_results = []
    if dist.get_rank() == 0:
        cache_paths = sorted(glob.glob(os.path.join(cache_path, "rank_*")), key=lambda x: int(x.split("rank_")[1]))
        results_for_each_file = [read_rawline_data(p) for p in cache_paths]
        while True:
            for sublist in results_for_each_file:
                if sublist:
                    if len(merged_results) >= len(input_lists):
                        break
                    merged_results.append(sublist.pop(0))
            if len(merged_results) >= len(input_lists) or all(not sublist for sublist in results_for_each_file):
                break
    return merged_results

# -------------------- MCTS --------------------
def distributed_inference_by_mcts(args, llm_dir, input_lists, override_cache=False, cache_suffix=""):
    cache_path = os.path.join(get_path(args, args.cache_dir), "cached_inference"+("_"+cache_suffix if cache_suffix else ""))
    if dist.get_rank() == 0:
        if override_cache:
            os.system(f"rm -rf {cache_path}")
        os.makedirs(cache_path, exist_ok=True)

    target_model_path = get_path(args, args.output_dir) if llm_dir is None else llm_dir    
    print_once(f">>> validate trg output_dir >>>:{target_model_path}")

    agent = TransAgent(args)
    sampler = torch.utils.data.distributed.DistributedSampler(input_lists, shuffle=False)
    data_loader = DataLoader(input_lists, shuffle=False, batch_size=args.per_device_eval_batch_size, sampler=sampler)

    dist_outs = []
    dist_preference_dfs = []
    progress_bar = tqdm(range(len(data_loader)), disable=(dist.get_rank() != 0))
    for _, batch_lines in enumerate(data_loader):
        progress_bar.update(1)
        for item in batch_lines:
            processed_line = item.split("<FUNC>")
            mc_tree = agent.MCTS(
                src_sent=processed_line[0],
                src_lang_code=processed_line[1],
                trg_lang_code=processed_line[2],
                show_info=False
            )
            item_list = mc_tree.layer_traversal(value_type="utility")
            root_data, root_value = item_list.pop(0)
            cleaned_dict = OrderedDict()
            for item_data, item_value in item_list:
                cleaned_dict.setdefault(item_data, []).append(item_value)
            cleaned_list = [(item_data, np.sum(item_value)) for item_data, item_value in cleaned_dict.items()]
            best_result = max(cleaned_list, key=lambda x: x[1])
            print(f"{root_data} ===> {best_result[0]}", end=" ")
            print(processed_line[2])
            dist_outs.append(best_result[0])
            df = agent.yield_tree2rank(mc_tree, threshold=root_value, value_type="utility")
            dist_preference_dfs.append(df)

    local_df = pd.concat(dist_preference_dfs, ignore_index=True)
    local_df.to_csv(os.path.join(cache_path, f"preference_{dist.get_rank()}.csv"), index=False)

    with open(os.path.join(cache_path, f"rank_{dist.get_rank()}"), "w") as cache_file:
        for l in dist_outs:
            cache_file.write(l + "\n")

    empty_cache()
    dist.barrier()

    merged_results = []
    if dist.get_rank() == 0:
        cache_paths = sorted(glob.glob(os.path.join(cache_path, "rank_*")), key=lambda x: int(x.split("rank_")[1]))
        results_for_each_file = [read_rawline_data(p) for p in cache_paths]
        while True:
            for sublist in results_for_each_file:
                if sublist:
                    if len(merged_results) >= len(input_lists):
                        break
                    merged_results.append(sublist.pop(0))
            if len(merged_results) >= len(input_lists) or all(not sublist for sublist in results_for_each_file):
                break

        df_path = glob.glob(os.path.join(cache_path, "preference_*"))
        collected_df = [pd.read_csv(f) for f in df_path]
        merged_df = pd.concat(collected_df, ignore_index=True)
        merged_df.fillna("", inplace=True)
        merged_df.to_csv(os.path.join(cache_path, "total_preference.csv"), index=False)

    print(">>merged_results>>:", merged_results)
    return merged_results
