from collections import defaultdict
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.accelerator import get_state_dict_offloaded_model
from accelerate.utils import get_balanced_memory
from compressed_tensors.quantization.utils import module_type
from compressed_tensors.utils import align_module_device
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor.modifiers import Modifier
from llmcompressor.pytorch.utils import get_linear_layers
from llmcompressor.pytorch.utils.helpers import tensor_sparsity

__ALL__ = [
    "calculate_offload_device_map",
    "tensor_follows_mask_structure",
    "infer_sparsity_structure_from_modifiers",
    "infer_sparsity_structure_from_model",
    "infer_sparse_targets_and_ignores",
    "is_sparse_compression_target",
]


def _resolve_torch_dtype(
    dtype: Optional[Union[str, torch.dtype]],
) -> Optional[torch.dtype]:
    if isinstance(dtype, torch.dtype):
        return dtype

    if isinstance(dtype, str):
        torch_dtype_str = dtype.replace("torch.", "").lower()
        if hasattr(torch, torch_dtype_str):
            resolved = getattr(torch, torch_dtype_str)
            if isinstance(resolved, torch.dtype):
                return resolved
    return None


def _estimate_cpu_available_bytes(default_gib: int = 128) -> int:
    try:
        import psutil  # type: ignore

        return max(int(psutil.virtual_memory().available), 1 * 1024**3)
    except Exception:
        return default_gib * 1024**3


def _build_max_memory_map(
    num_gpus: int, reserve_for_hessians: bool
) -> Dict[str, int]:
    max_memory = {}
    if num_gpus <= 0 or not torch.cuda.is_available():
        return max_memory

    reserve_ratio = 0.2 if reserve_for_hessians else 0.0
    min_reserve_bytes = 2 * 1024**3 if reserve_for_hessians else 0

    for gpu_idx in range(num_gpus):
        try:
            _, total_bytes = torch.cuda.mem_get_info(gpu_idx)
        except RuntimeError:
            total_bytes = torch.cuda.get_device_properties(gpu_idx).total_memory

        reserved_bytes = max(int(total_bytes * reserve_ratio), min_reserve_bytes)
        available_bytes = max(total_bytes - reserved_bytes, total_bytes // 2)
        # Clamp to at least 1 GiB to avoid zero allocations
        available_bytes = max(available_bytes, 1 * 1024**3)

        max_memory[f"cuda:{gpu_idx}"] = available_bytes

    return max_memory


def calculate_offload_device_map(
    model_id: str,
    *,
    num_gpus: Optional[int] = None,
    reserve_for_hessians: bool = False,
    torch_dtype: Optional[Union[str, torch.dtype]] = None,
    trust_remote_code: bool = False,
    allow_cpu_offload: bool = True,
) -> Union[str, Dict[str, int]]:
    """Infer a device map that balances model shards across accelerators.

    Falls back to CPU execution when no CUDA devices are available. When GPUs
    are present, the helper reserves a portion of device memory to accommodate
    additional buffers (e.g. Hessians) before delegating to Accelerate's
    heuristics for sharding.

    :param model_id: HuggingFace model identifier or local path.
    :param num_gpus: Optional cap on the number of CUDA devices to consider.
    :param reserve_for_hessians: If True, reserves ~20% (at least 2 GiB) of each
        GPU's memory to reduce OOM risk during Hessian accumulation.
    :param torch_dtype: Optional dtype override used for memory estimation.
    :param trust_remote_code: Forwarded to HuggingFace loaders for custom models.
    :param allow_cpu_offload: Whether to include CPU in memory map for offloading.
    :return: A device map consumable by `from_pretrained`, or "cpu" when no GPU
        resources are detected.
    """

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus is None:
        num_gpus = available_gpus
    else:
        if num_gpus > available_gpus:
            warnings.warn(
                (
                    "Requested num_gpus=%s but only %s CUDA devices are visible. "
                    "Reducing to the available count."
                )
                % (num_gpus, available_gpus)
            )
            num_gpus = available_gpus

    if num_gpus == 0:
        return "cpu"

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=trust_remote_code
    )

    resolved_dtype = (
        _resolve_torch_dtype(torch_dtype)
        or _resolve_torch_dtype(getattr(config, "torch_dtype", None))
        or (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        )

    no_split_module_classes = model._get_no_split_modules("auto")

    max_memory = _build_max_memory_map(num_gpus, reserve_for_hessians)
    if allow_cpu_offload:
        max_memory.setdefault("cpu", _estimate_cpu_available_bytes())

    try:
        balanced_memory = get_balanced_memory(
            model,
            max_memory=max_memory if max_memory else None,
            no_split_module_classes=no_split_module_classes,
            dtype=resolved_dtype,
        )
        device_map = infer_auto_device_map(
            model,
            max_memory=balanced_memory if balanced_memory else None,
            no_split_module_classes=no_split_module_classes,
            dtype=resolved_dtype,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        warnings.warn(
            "Falling back to the default Accelerate heuristics for device map "
            f"inference because automatic balancing failed: {exc}"
        )
        device_map = "auto"
    finally:
        # Ensure temporary meta model does not linger longer than needed
        del model

    return device_map


def tensor_follows_mask_structure(tensor: torch.Tensor, mask: str = "2:4") -> bool:
    """
    :param tensor: tensor to check
    :param mask: mask structure to check for, in the format "n:m", also accepts
        "unstructured" as a valid mask structure
    :return: True if the tensor follows the mask structure, False otherwise.
        Note, some weights can incidentally be zero, so we check for
        atleast n zeros in each chunk of size m
    """

    if mask.lower().strip() == "unstructured":
        return True

    n, m = tuple(map(int, mask.split(":")))

    # If n or m is 0, then the tensor follows the mask structure
    if n == 0 or m == 0:
        return True
    # Reshape the tensor into chunks of size m
    tensor = tensor.view(-1, m)

    # Count the number of zeros in each chunk
    zero_counts = (tensor == 0).sum(dim=1)

    # Check if the number of zeros in each chunk atleast n
    # Greater than sign is needed as some weights can incidentally
    # be zero
    return torch.all(zero_counts >= n).item()


def infer_sparsity_structure_from_modifiers(
    modifiers: List[Modifier],  # noqa E501
) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the list of modifiers.

    :param modifiers: List of modifier instances.
    :return: sparsity structure as a string or None.
    """
    for modifier in modifiers:
        if hasattr(modifier, "mask_structure"):
            return modifier.mask_structure
    return None


def infer_sparsity_structure_from_model(model: torch.nn.Module) -> Optional[str]:
    """
    Determines the sparsity structure, if any exists, given the model

    :param model: model to check for sparsity structure
    :return: sparsity structure as a string or None
    """

    # check for the common sparsity structures
    structures = {"2:4"}
    for sparsity_structure in structures:
        linear_modules = get_linear_layers(model)
        offloaded_params = get_state_dict_offloaded_model(model)

        linear_modules_with_sparsity_structure = [
            tensor_follows_mask_structure(offloaded_params[f"{name}.weight"])
            for name in tqdm(
                linear_modules.keys(),
                desc="Checking whether model follows "
                f"{sparsity_structure} sparsity structure",
            )
        ]
        # if the majority of the linear modules follow the sparsity structure
        # we can assume that the model follows the sparsity structure
        # (taking into consideration the fact that some Linear layers like the
        # embedding layer might not be sparse)
        if (
            sum(linear_modules_with_sparsity_structure)
            > len(linear_modules_with_sparsity_structure) * 0.8
        ):
            return sparsity_structure

    return None


def infer_sparse_targets_and_ignores(
    model: torch.nn.Module,
    sparsity_structure: str,
    sparsity_threshold: float,
) -> Tuple[List[str], List[str]]:
    """
    Infers the target and ignore layers in the given model
    to be used for sparsity compression

    :param model: model to check
    :param sparsity_structure: sparsity structure to check against
    :param sparsity_threshold: threshold for sparsity
    :return: tuple of target and ignore layers
    """

    exhaustive_targets, exhaustive_ignore = _get_sparse_targets_ignore_dicts(
        module=model,
        sparsity_structure=sparsity_structure,
        sparsity_threshold=sparsity_threshold,
    )

    return _reduce_targets_and_ignores_into_lists(
        exhaustive_targets=exhaustive_targets,
        exhaustive_ignore=exhaustive_ignore,
    )


def is_sparse_compression_target(
    module: torch.nn.Module, sparsity_threshold: float, sparsity_structure: str
) -> bool:
    """
    :param module: module to check
    :param sparsity_threshold: threshold for sparsity
    :param sparsity_structure: sparsity structure to check against
    :return: whether or not the module is a target for sparsity compression,
        i.e True if it is sparse and follows the sparsity structure, else False
    """
    with align_module_device(module):
        result = (
            hasattr(module, "weight")
            and tensor_sparsity(module.weight) >= sparsity_threshold
            and tensor_follows_mask_structure(
                tensor=module.weight, mask=sparsity_structure
            )
        )

    return result


def _get_sparse_targets_ignore_dicts(
    module: torch.nn.Module, sparsity_structure: str, sparsity_threshold: float
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Get sparse targets and ignore dictionaries

    :param module: module to check
    :param sparsity_structure: sparsity structure to check against
    :param sparsity_threshold: threshold for sparsity
    :return: tuple of exhaustive targets and ignore dictionaries
    """
    exhaustive_targets = defaultdict(list)
    exhaustive_ignore = defaultdict(list)

    for name, submodule in module.named_modules():
        submodule_type = module_type(submodule)
        is_target = is_sparse_compression_target(
            module=submodule,
            sparsity_threshold=sparsity_threshold,
            sparsity_structure=sparsity_structure,
        )
        target_dict = exhaustive_targets if is_target else exhaustive_ignore
        target_dict[submodule_type].append(name)
    return exhaustive_targets, exhaustive_ignore


def _reduce_targets_and_ignores_into_lists(
    exhaustive_targets: Dict[str, List[str]], exhaustive_ignore: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    """
    Reduces the targets and ignores dictionaries into lists

    :param exhaustive_targets: dictionary of target layers, must contain all
        targetted layers in the model
    :param exhaustive_ignore: dictionary of ignore layers, must contain all
        ignored layers in the model
    :return: tuple of reduced target and ignore layers
    """

    targets, ignore = [], []
    all_submodule_types = set(exhaustive_targets.keys()).union(
        set(exhaustive_ignore.keys())
    )
    for submodule_type in all_submodule_types:
        curr_targets = exhaustive_targets.get(submodule_type, [])
        curr_ignores = exhaustive_ignore.get(submodule_type, [])

        if len(curr_targets) >= len(curr_ignores):
            targets.append(submodule_type)
            ignore.extend(curr_ignores)
        elif len(curr_targets) > 0:
            # only add ignore layers if
            # they are targetted
            targets.extend(curr_targets)
            ignore.extend(curr_ignores)
    return targets, ignore
