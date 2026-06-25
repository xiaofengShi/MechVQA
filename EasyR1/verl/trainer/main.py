# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# limitations under the License.
import json
import os

try:
    import weave  # noqa: F401
except ImportError:
    weave = None

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .config import PPOConfig
from .data_loader import create_dataloader
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
            Role.Critic: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config: PPOConfig = OmegaConf.to_object(ppo_config)
    ppo_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
                # 跨节点 vLLM 通信
                "VLLM_HOST_IP": os.environ.get("VLLM_HOST_IP", "127.0.0.1"),
                "MASTER_ADDR": os.environ.get("MASTER_ADDR", "127.0.0.1"),
                "MASTER_PORT": os.environ.get("MASTER_PORT", "29500"),
                "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME", "bond0"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "bond0"),
                "GLOO_USE_IPV6": "0",
                "NCCL_NVLS_ENABLE": "0",
                "NCCL_P2P_LEVEL": "NVL",
                "NCCL_SHARP_DISABLE": "1",
                "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
                "WANDB_DIR": os.environ.get("WANDB_DIR", "wandb"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
                "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE", "1"),
                "TRITON_CACHE_DIR": "/tmp/triton_local",
                "LD_PRELOAD": "/usr/local/cuda/compat/lib/libcuda.so.1",
                "ENABLE_THINKING": os.environ.get("ENABLE_THINKING", "true"),
                "HF_HOME": os.environ.get("HF_HOME", ""),
                "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", ""),
                "HF_ENDPOINT": os.environ.get("HF_ENDPOINT", ""),
            }
        }
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            ray.init(address=ray_address, runtime_env=runtime_env)
        else:
            ray.init(runtime_env=runtime_env)

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))

    if ppo_config.trainer.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=ppo_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
