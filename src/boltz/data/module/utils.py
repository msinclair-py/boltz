"""Utility functions for data modules."""

import torch

try:
    import intel_extension_for_pytorch
except ImportError:
    pass



# Keys that should NOT be transferred to device (non-tensor or special fields)
_SKIP_KEYS_V1 = frozenset([
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
])

_SKIP_KEYS_V2 = frozenset([
    "all_coords",
    "all_resolved_mask",
    "crop_to_all_atom_map",
    "chain_symmetries",
    "amino_acids_symmetries",
    "ligand_symmetries",
    "record",
    "affinity_mw",
])


def transfer_batch_to_device(
    batch: dict,
    device: torch.device,
    skip_keys: frozenset = _SKIP_KEYS_V2,
) -> dict:
    """Transfer a batch to the given device.

    Iterates over top-level keys and moves tensors to device,
    skipping keys in the skip_keys set (non-tensor or special fields).

    Parameters
    ----------
    batch : dict
        The batch to transfer.
    device : torch.device
        The device to transfer to.
    skip_keys : frozenset
        Set of keys to skip (not transfer to device).
        Defaults to _SKIP_KEYS_V2 which is the superset of all skip keys.

    Returns
    -------
    dict
        The transferred batch.

    """
    for key in batch:
        if key not in skip_keys:
            batch[key] = batch[key].to(device)
    return batch
