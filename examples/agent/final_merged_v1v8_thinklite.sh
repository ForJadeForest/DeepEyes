set -x

export WANDB_API_KEY=070da1234f26b6663537b7c68d1084b0eaecd342
export WANDB_ENTITY="ucasrhk-ucas"
export WANDB_PROJECT="DeepEye-Yingzhe"
wandb login

PROJECT_NAME="DeepEye-Yingzhe"
EXPERIMENT_NAME="Qwen2.5-7B-DeepEyes-Baseline"

# the IP and port for your Qwen-2.5-72B-Instruct vllm serving
export LLM_AS_A_JUDGE_BASE="http://29.191.208.124:18901/v1"

# number of training nodes
export WORLD_SIZE=6

export SAVE_CHECKPOINT_DIR="/apdcephfs_gy5/share_303588738/yingzhepeng/results/deepeye"


mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/train_logs
mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/logs

# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

export PATH="/jizhicfs/berlinni/miniconda3/envs/deepeye_pyz/bin:$PATH"
export LD_LIBRARY_PATH="/jizhicfs/berlinni/miniconda3/envs/deepeye_pyz/lib/python3.10/site-packages/nvidia/cublas/lib/:/jizhicfs/berlinni/miniconda3/envs/deepeye_pyz/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH"
export NCCL_IB_GID_INDEX=3 
export NCCL_IB_SL=3 
export NCCL_CHECKS_DISABLE=1 
export NCCL_P2P_DISABLE=0 
export NCCL_IB_DISABLE=0 
export NCCL_LL_THRESHOLD=16384 
export NCCL_IB_CUDA_SUPPORT=1 
export NCCL_SOCKET_IFNAME=bond1 
export UCX_NET_DEVICES=bond1 
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6 
export NCCL_COLLNET_ENABLE=0 
export SHARP_COLL_ENABLE_SAT=0 
export NCCL_NET_GDR_LEVEL=2 
export NCCL_IB_QPS_PER_CONNECTION=4 
export NCCL_IB_TC=160 
export NCCL_PXN_DISABLE=0 
export NCCL_DEBUG=INFO       # 输出调试信息

export https_proxy=http://star-proxy.oa.com:3128
export http_proxy=http://star-proxy.oa.com:3128

BASEDIR=/apdcephfs_sh8/share_301266059/berlinni/shihoukun/deepeyes/DeepEyes-Datasets-47k
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet
VSATR_VAL=${BASEDIR}/sample_val.parquet

REF_MODEL_PATH=/apdcephfs_sh8/share_301266059/berlinni/public_ckpts/Qwen2.5-VL-7B-Instruct
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}] \
    data.train_batch_size=384 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=384 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=4096 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=5 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    reward_model.reward_manager=prime \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=30 \
    trainer.test_freq=5 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.log_val_generations=15 \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/logs/rl_logging_board \
    trainer.total_epochs=1 2>&1 | tee ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/train_logs/${EXPERIMENT_NAME}.log