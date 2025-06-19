set -x

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="try_all_merged_v32"

export SAVE_CHECKPOINT_DIR="/apdcephfs_gy5/share_303588738/yingzhepeng/results/deepeye"


mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/train_logs
mkdir -p ${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}/logs

# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

BASEDIR=/cpfs/user/fengyuan/verl_data/minghao_data
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/minghao_data_vnew/data_v0.8_visual_toolbox_v2.parquet
VISUAL_DATASET_TEST=${BASEDIR}/seekworld_test.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_function_call.parquet
XINCE_DATASET_TRAIN=${BASEDIR}/train_xince_acc.parquet
SEEKWORLD_DATASET_TRAIN=${BASEDIR}/seekworld_train_acc.parquet

DATA_V2_TRAIN_0_1_2=/cpfs/user/fengyuan/verl_data/minghao_data/data_0.1.2_visual_toolbox_v2_acc.parquet
DATA_V2_TRAIN_0_8_SPLIT1=/cpfs/user/fengyuan/verl_data/minghao_data/minghao_data_vnew/data_v0.8_visual_toolbox_v2_acc_split1.parquet
DATA_V2_TRAIN_0_8_SPLIT2=/cpfs/user/fengyuan/verl_data/minghao_data/minghao_data_vnew/data_v0.8_visual_toolbox_v2_acc_split2.parquet
DATA_V2_TRAIN_THINKLITE=/cpfs/user/fengyuan/verl_data/minghao_data/data_thinklite_reasoning_function_call_acc.parquet
DATA_V2_TRAIN_XINCE=/cpfs/user/fengyuan/verl_data/minghao_data/train_xince_acc_acc.parquet
DATA_V2_TRAIN_GEOGUESSR=/cpfs/user/fengyuan/verl_data/minghao_data/seekworld_train_acc_acc.parquet

REF_MODEL_PATH=/apdcephfs_sh8/share_301266059/berlinni/public_ckpts/Qwen2.5-VL-7B-Instruct
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files=[${DATA_V2_TRAIN_0_1_2},${DATA_V2_TRAIN_0_8_SPLIT1},${DATA_V2_TRAIN_0_8_SPLIT2},${DATA_V2_TRAIN_THINKLITE},${DATA_V2_TRAIN_XINCE},${DATA_V2_TRAIN_GEOGUESSR}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}] \
    data.train_batch_size=384 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=8192 \
    actor_rollout_ref.rollout.agent.max_turns=20 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
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