"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding); a variable fraction of tokens is cropped at row
     boundaries, since whatever's left over after the best-fitting docs are placed
     gets filled by cropping the shortest buffered doc down to the exact remainder

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses some tokens to cropping, but ensures that there are fewer
"confusing" tokens in the train/val batches as every token can now attend
back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117

`BestFitPacker` below is the packing algorithm shared by the live training
loader and the offline row-packing pipeline in nanochat/pack.py (which
pre-packs rows to disk so the wiki-retrieval encoder has a fixed reference
target to build against; see nanochat/wiki.py).
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size, rank=None, world_size=None):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).

    rank/world_size default to the current DDP rank/world size, but can be overridden together
    (e.g. by the offline packer, which runs several independent streams outside of DDP).
    """
    if rank is None:
        _, rank, _, world_size = get_dist_info()

    warn_on_legacy = rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * world_size + rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


class BestFitPacker:
    """
    Best-fit document packer. Repeatedly refills a buffer of tokenized documents from
    `batches` and packs rows of exactly `row_capacity` tokens:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop the SHORTEST buffered doc to fill the remaining space exactly

    Every row starts with BOS (each document in the buffer was BOS-prepended at tokenize
    time), and every row is 100% utilized (no padding, every token is trained on).
    """
    def __init__(self, batches, tokenizer, tokenizer_threads=4, buffer_size=1000):
        self.batches = batches
        self.tokenizer = tokenizer
        self.bos_token = tokenizer.get_bos_token_id()
        self.tokenizer_threads = tokenizer_threads
        self.buffer_size = buffer_size
        self.doc_buffer = []
        self.pq_idx, self.rg_idx, self.epoch = 0, 0, 1

    def _refill(self):
        doc_batch, (self.pq_idx, self.rg_idx, self.epoch) = next(self.batches)
        token_lists = self.tokenizer.encode(doc_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads)
        self.doc_buffer.extend(token_lists)

    def pack_row(self, row_capacity):
        """ Fill one row of exactly row_capacity tokens via best-fit packing. Returns list[int]. """
        row = []
        pos = 0
        while pos < row_capacity:
            while len(self.doc_buffer) < self.buffer_size:
                self._refill()
            remaining = row_capacity - pos

            best_idx, best_len = -1, 0
            for i, doc in enumerate(self.doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx, best_len = i, doc_len

            if best_idx >= 0:
                doc = self.doc_buffer.pop(best_idx)
                row.extend(doc)
                pos += len(doc)
            else:
                # Nothing fits - crop shortest in buffer to fill remaining and minimize waste
                shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                doc = self.doc_buffer.pop(shortest_idx)
                row.extend(doc[:remaining])
                pos += remaining
        return row

    @property
    def state_dict(self):
        return {"pq_idx": self.pq_idx, "rg_idx": self.rg_idx, "epoch": self.epoch}


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """ BOS-aligned dataloader with best-fit packing. See BestFitPacker for the algorithm. """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    packer = BestFitPacker(batches, tokenizer, tokenizer_threads=tokenizer_threads, buffer_size=buffer_size)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            row = packer.pack_row(row_capacity)
            row_t = torch.tensor(row, dtype=torch.long)
            cpu_inputs[row_idx] = row_t[:-1]
            cpu_targets[row_idx] = row_t[1:]

        state_dict = packer.state_dict

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

# -----------------------------------------------------------------------------
# Pre-packed rows (reference/"cheat sheet" experiment)

def packed_data_loader_with_state(B, T, split, device="cuda", resume_state_dict=None):
    """
    Dataloader over the offline pre-packed rows of nanochat/pack.py, whose wiki reference
    columns were filled in by scripts/wiki_retrieve.py. Yields (inputs, targets, refs, state_dict)
    where inputs/targets are (B, T) and refs are (B, n_ref, ref_len).

    pack.py interleaves its streams so that file row i*world_size+r is stream r's i-th row,
    so rank r simply walks its own stream in order. The set of rows an optimizer step sees
    therefore depends only on the world size and total batch size, never on device_batch_size,
    which is what lets two arms with different memory footprints train on identical data.
    Resume is exact: state_dict is the stream position of the batch being yielded.
    """
    import numpy as np
    from nanochat.pack import packed_path, DOC_WIDTH, NUM_WINDOWS, WINDOW_LEN # (lazy: nanochat.pack imports this module)
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert T + 1 == DOC_WIDTH, f"packed rows hold {DOC_WIDTH} doc tokens, so T must be {DOC_WIDTH - 1} (got {T})"

    _, rank, _, world_size = get_dist_info()
    rows = np.load(packed_path(split), mmap_mode="r")
    stream_len = rows.shape[0] // world_size
    assert stream_len >= B, f"{packed_path(split)} has only {stream_len} rows per rank, need at least {B}"
    pos = resume_state_dict["pos"] if resume_state_dict is not None else 0
    epoch = resume_state_dict["epoch"] if resume_state_dict is not None else 1

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T) | refs (B*P)]
    P = NUM_WINDOWS * WINDOW_LEN
    use_cuda = device == "cuda"
    cpu_buffer = torch.empty(2 * B * T + B * P, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T + B * P, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:2 * B * T].view(B, T)
    cpu_refs = cpu_buffer[2 * B * T:].view(B, P)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:2 * B * T].view(B, T)
    refs = gpu_buffer[2 * B * T:].view(B, NUM_WINDOWS, WINDOW_LEN)

    copy_done = None
    while True:
        if pos + B > stream_len:
            pos, epoch = 0, epoch + 1 # wrap around (drops the stream's last partial batch)
        state_dict = {"pos": pos, "epoch": epoch}
        batch = torch.from_numpy(rows[pos * world_size + rank:(pos + B) * world_size:world_size].astype(np.int64))
        # don't overwrite the pinned staging buffer while the previous HtoD copy may still be reading it
        if copy_done is not None:
            copy_done.synchronize()
        cpu_inputs.copy_(batch[:, :T])
        cpu_targets.copy_(batch[:, 1:DOC_WIDTH])
        cpu_refs.copy_(batch[:, DOC_WIDTH:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        if use_cuda:
            copy_done = torch.cuda.Event()
            copy_done.record()
        pos += B
        yield inputs, targets, refs, state_dict

def packed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, refs, state_dict in packed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets, refs
