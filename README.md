# NPU Environment Setup

## Hardware Specifications

| Hardware Model              | 910B3 (Recommended) |
| --------------------------- | ------------------- |
| Software Version            | 24.1.rc3            |
| Driver Version              | 24.1.rc3            |
| Firmware Version            | 7.8.0.2.212         |
| Ascend CANN Toolkit Version | 8.3.RC2             |

### Environment Preparation

```bash
# Set versions
npu="910b"
drv_version="24.1.RC3"
d_version="24.1.rc3"
firmware_ver="7.8.0.2.212"
cann_ver="8.3.RC2"
arch=$(uname -i)

# Install NPU driver
drv="Ascend-hdk-${npu}-npu-driver_${d_version}_linux-aarch64.run"
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%20${drv_version}/${drv}"
bash "$drv" --full --install-for-all

# Install firmware
fw="Ascend-hdk-${npu}-npu-firmware_${firmware_ver}.run"
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%20${drv_version}/${fw}"
bash "$fw" --full

# Install CANN toolkit
tk="Ascend-cann-toolkit_${cann_ver}_linux-${arch}.run"
wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${cann_ver}/${tk}"
bash "$tk" --full
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install kernels
kernels="Ascend-cann-kernels-${npu}_${cann_ver}_linux-${arch}.run"
wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${cann_ver}/${kernels}"
bash "$kernels" --install

# Install NNAL
nnal="Ascend-cann-nnal_${cann_ver}_linux-${arch}.run"
wget --header="Referer: https://www.hiascend.com/" "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${cann_ver}/${nnal}"
bash "$nnal" --install
source /usr/local/Ascend/nnal/atb/set_env.sh

echo "Ascend NPU (${npu}) environment setup completed."
```

---

## Key Software Versions (vLLM)

| Software    | Version   |
| ----------- | --------- |
| vllm        | 0.13.0    |
| vllm-ascend | 0.13.0rc1 |

Reference: [vLLM Ascend Installation Guide](https://docs.vllm.ai/projects/ascend/zh-cn/v0.13.0/installation.html)
A full Python environment example is provided in [env.yaml](./env.yaml).

### Installation Steps

```bash
# Install vLLM
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# Install vLLM Ascend
git clone --depth 1 --branch v0.13.0rc2 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e .
cd ..
```

> ⚠️ **Important:** Use **Python 3.10** for best compatibility.
>
> * **Step 1:** Install the **NPU-compatible vLLM** first.
> * **Step 2:** Install all additional Python dependencies with:
>
>   ```bash
>   pip install -r requirements-npu.txt
>   ```
>
> ⚠️ **Performance Notice:**
> The **vLLM library is the main bottleneck** for NPU performance.
> Multi-GPU setups **have not been tested** and may require further optimization.

---

## Running Examples

To help users quickly run the code, a sample [dataset](./dataset) is provided:

* **SFT**: [dataset/sft](./dataset/sft) contains example bilingual data in `jsonl` format for SFT training.
* **Monolingual**: [dataset/monolingual](./dataset/monolingual) contains data needed for self-play stages.

---

## Pre-download Evaluation Models

Before running evaluation, you need to prepare models for translation quality assessment.

### Download the models

```bash
# Create necessary folders
mkdir -p ./models/huggingface ./models/Unbabel

# Download BLEURT-20
hf download lucadiliello/BLEURT-20 --local-dir ./models/huggingface/bleurt20

# Download COMET-Kiwi (Unbabel)
hf download Unbabel/wmt22-cometkiwi-da --local-dir ./models/Unbabel/wmt22-cometkiwi-da
```

### Directory Structure

After downloading, the folder structure should look like this:

```
main.py
models/
├── huggingface/
│   └── bleurt20/
└── Unbabel/
    └── wmt22-cometkiwi-da/
```

> ⚠️ **Note:** Ensure the directories exist exactly as shown; otherwise evaluation scripts may fail to locate the models.

---

## Quick Run Examples (Single-Card)

```bash
# SFT Training
torchrun \
  --nproc_per_node=1 \
  main.py \
  --mode SFT \
  --llm_path /path/to/models/Qwen3-0.6B \
  --train_data_path /path/to/data/sft \
  --cache_dir cache/qwen_debug/ \
  --output_dir ckpts/qwen_debug/ \
  --deepspeed configs/ds_z2_config.json \
  --learning_rate 3e-6 \
  --run_name qwen_baseline \
  --report_to none \
  --bf16 \
  2>&1 | tee continue.log

# RL Training
ARNOLD_WORKER_NUM=1 torchrun \
  --nproc_per_node=1 \
  main.py \
  --mode RL \
  --mcts_sample_size 1 \
  --mc_count 2 \
  --train_rounds 2 \
  --llm_path /path/to/models/Qwen3-0.6B \
  --train_data_path ./dataset/sft \
  --src_code eng_Latn \
  --trg_code zho_Hans \
  --self_play_languages "eng_Latn" "zho_Hans" "deu_Latn" \
  --cache_dir cache/qwen_debug/ \
  --flores_script flores200.py \
  --output_dir ckpts/qwen_debug/ \
  --nas_base_path . \
  --deepspeed configs/ds_z2_config.json \
  --rl_loss_type sppo_hard \
  --learning_rate 1e-4 \
  --rl_learning_rate 1e-6 \
  --run_name qwen_rl_debug \
  --report_to none \
  2>&1 | tee continue.log
```
