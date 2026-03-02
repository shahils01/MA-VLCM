import io
import os
import fnmatch
import re
from collections import defaultdict, deque

import torch
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds


def _as_numpy(x):
    if isinstance(x, bytes):
        import numpy as np
        try:
            return np.load(io.BytesIO(x), allow_pickle=True)
        except Exception:
            return torch.load(io.BytesIO(x), map_location="cpu")
    return x


def _is_glob_pattern(s):
    return any(ch in s for ch in ("*", "?", "["))


def _hf_list_dataset_files(repo_id, revision):
    try:
        from huggingface_hub import HfApi
    except Exception as e:
        raise RuntimeError(
            "Hugging Face wildcard shard expansion requires `huggingface_hub`. "
            "Install it with: pip install huggingface_hub"
        ) from e
    api = HfApi()
    return api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)


def _expand_hf_uri_spec(spec):
    # Format: hf://datasets/<org>/<repo>@<revision>/<path_or_glob>
    m = re.match(r"^hf://datasets/([^/@]+/[^/@]+)(?:@([^/]+))?/(.+)$", spec)
    if not m:
        return [spec]
    repo_id, revision, repo_path = m.group(1), (m.group(2) or "main"), m.group(3)
    if not _is_glob_pattern(repo_path):
        return [f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{repo_path}"]

    files = _hf_list_dataset_files(repo_id, revision)
    matches = [f for f in files if fnmatch.fnmatch(f, repo_path)]
    matches = sorted([f for f in matches if f.endswith(".tar")])
    if not matches:
        raise RuntimeError(f"No .tar files matched '{spec}'")
    return [f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{f}" for f in matches]


def _expand_hf_http_spec(spec):
    # Format: https://huggingface.co/datasets/<org>/<repo>/tree/<revision>/<path_or_glob>
    # Also supports /resolve/<revision>/...
    m = re.match(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/(tree|resolve)/([^/]+)/(.+)$", spec)
    if not m:
        return [spec]
    repo_id, mode, revision, repo_path = m.group(1), m.group(2), m.group(3), m.group(4)

    if not _is_glob_pattern(repo_path):
        if mode == "tree":
            return [f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{repo_path}"]
        return [spec]

    files = _hf_list_dataset_files(repo_id, revision)
    matches = [f for f in files if fnmatch.fnmatch(f, repo_path)]
    matches = sorted([f for f in matches if f.endswith(".tar")])
    if not matches:
        raise RuntimeError(f"No .tar files matched '{spec}'")
    return [f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{f}" for f in matches]


def resolve_shards_spec(shards):
    parts = [p.strip() for p in str(shards).split("::") if p.strip()]
    resolved = []
    for p in parts:
        if p.startswith("hf://datasets/"):
            resolved.extend(_expand_hf_uri_spec(p))
        elif p.startswith("https://huggingface.co/datasets/"):
            resolved.extend(_expand_hf_http_spec(p))
        else:
            resolved.append(p)
    return "::".join(resolved)


def _edge_index_to_adj(edge_index, num_nodes):
    edge_index = _as_numpy(edge_index)
    if torch.is_tensor(edge_index):
        ei = edge_index
    else:
        import numpy as np
        ei = torch.as_tensor(np.asarray(edge_index))

    if ei.dim() == 2 and ei.shape[0] == num_nodes and ei.shape[1] == num_nodes:
        return ei.float()
    if ei.dim() == 2 and ei.shape[0] == 2:
        src = ei[0].long()
        dst = ei[1].long()
        mask = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
        src = src[mask]
        dst = dst[mask]
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        if src.numel() > 0:
            adj[src, dst] = 1.0
        return adj
    raise ValueError("edge_index has unexpected shape")


def _reduce_value(x, reduce_mode):
    if reduce_mode == "sum":
        return x.sum()
    if reduce_mode == "first":
        return x.flatten()[0]
    return x.mean()


def _reduce_done(x, reduce_mode):
    if reduce_mode == "all":
        return (x > 0).all()
    if reduce_mode == "sum":
        return x.sum() > 0
    if reduce_mode == "first":
        return x.flatten()[0] > 0
    if reduce_mode == "mean":
        return x.float().mean() > 0.5
    return (x > 0).any()


def _reward_from_frame(sample, reduce_mode):
    arr = _as_numpy(sample["rewards.npy"])
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    t = torch.tensor(arr, dtype=torch.float32)
    return _reduce_value(t, reduce_mode)


def _done_from_frame(sample, reduce_mode):
    arr = _as_numpy(sample["dones.npy"])
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    t = torch.tensor(arr, dtype=torch.float32)
    return _reduce_done(t, reduce_mode)


def _extract_episode_id(key):
    key = str(key)
    if "_step_" in key:
        return key.split("_step_", 1)[0]
    if "_" in key:
        return key.rsplit("_", 1)[0]
    return key


def preprocess_vlm_video_inputs(
    vlm_processor,
    frames,
    text,
    text_prompt_template=None,
    vlm_max_text_len=256,
    vlm_truncation=False,
    vlm_padding="longest",
    squeeze_batch_dim=True,
):
    if vlm_processor is None:
        raise RuntimeError("Dataloader processor not set.")
    if not isinstance(text, str):
        text = text_prompt_template
    if not isinstance(text, str):
        text = ""
    proc_text = text

    tokenizer = getattr(vlm_processor, "tokenizer", None)
    if tokenizer is not None:
        vocab = tokenizer.get_vocab()
        if "<video>" in vocab and "<video>" not in proc_text and "<image>" not in proc_text:
            proc_text = f"<video>\n{proc_text}"
        if "<obs>" in vocab and "<obs>" not in proc_text:
            if "<video>" in proc_text:
                proc_text = proc_text.replace("<video>\n", "<video><obs>\n", 1)
            else:
                proc_text = f"<obs>\n{proc_text}"

    # For batched videos (list of list-of-frames), HF processors expect batched text.
    if isinstance(frames, (list, tuple)) and len(frames) > 0 and isinstance(frames[0], (list, tuple)):
        proc_text = [proc_text for _ in range(len(frames))]

    try:
        max_len = vlm_max_text_len if vlm_truncation else None
        inputs = vlm_processor(
            text=proc_text,
            videos=frames,
            return_tensors="pt",
            padding=vlm_padding,
            truncation=vlm_truncation,
            max_length=max_len,
        )
    except TypeError:
        inputs = vlm_processor(images=frames, return_tensors="pt")

    packed = {}
    for k, v in dict(inputs).items():
        if squeeze_batch_dim and torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
            v = v.squeeze(0)
        packed[k] = v
    return packed


class SequenceWebDataset(IterableDataset):
    def __init__(
        self,
        shards,
        clip_len,
        clip_stride,
        text_mode,
        robot_source,
        reward_reduce,
        done_reduce,
        vlm_processor=None,
        text_prompt_template=None,
        include_next=False,
        vlm_max_text_len=256,
        vlm_truncation=False,
        vlm_padding="longest",
        gamma=0.99,
        return_horizon="clip",
        n_step=1,
        include_nstep_bootstrap=False,
        num_robots=None,
        robot_obs_dim=None,
    ):
        self.shards = shards
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.text_mode = text_mode
        self.robot_source = robot_source
        self.reward_reduce = reward_reduce
        self.done_reduce = done_reduce
        self.vlm_processor = vlm_processor
        self.text_prompt_template = text_prompt_template
        self.include_next = include_next
        self.vlm_max_text_len = vlm_max_text_len
        self.vlm_truncation = vlm_truncation
        self.vlm_padding = vlm_padding
        self.gamma = float(gamma)
        self.n_step = max(1, int(n_step))
        self.include_nstep_bootstrap = include_nstep_bootstrap
        self.num_robots = int(num_robots) if num_robots is not None else None
        self.robot_obs_dim = int(robot_obs_dim) if robot_obs_dim is not None else None
        if return_horizon not in {"clip", "trajectory"}:
            raise ValueError("return_horizon must be one of {'clip', 'trajectory'}")
        self.return_horizon = return_horizon

    @staticmethod
    def _as_bool(x):
        if torch.is_tensor(x):
            if x.numel() == 0:
                return False
            return bool(x.detach().float().max().item() > 0.0)
        return bool(x)

    @staticmethod
    def _terminal_pad_from(frame):
        padded = dict(frame)
        reward = frame["reward"]
        done = frame["done"]
        if torch.is_tensor(reward):
            padded["reward"] = reward.new_zeros(())
        else:
            padded["reward"] = 0.0
        if torch.is_tensor(done):
            padded["done"] = torch.ones_like(done, dtype=done.dtype)
        else:
            padded["done"] = True
        return padded

    def _apply_done_termination(self, clip):
        done_idx = None
        for idx, frame in enumerate(clip):
            if self._as_bool(frame["done"]):
                done_idx = idx
                break
        if done_idx is None or done_idx == len(clip) - 1:
            return clip

        terminal = clip[done_idx]
        out = list(clip[: done_idx + 1])
        for _ in range(done_idx + 1, len(clip)):
            out.append(self._terminal_pad_from(terminal))
        return out

    def _discounted_return(self, frames):
        ret = None
        discount = 1.0
        for frame in frames:
            reward = frame["reward"]
            if ret is None:
                ret = torch.zeros_like(reward)
            ret = ret + discount * reward
            if self._as_bool(frame["done"]):
                break
            discount *= self.gamma
        if ret is None:
            ret = torch.tensor(0.0, dtype=torch.float32)
        return ret

    def _nstep_discounted_return(self, buffer, clip_start):
        reward_start = clip_start + self.clip_len - 1
        ret = None
        discount = 1.0
        done_n = False
        for j in range(self.n_step):
            idx = reward_start + j
            if idx >= len(buffer):
                done_n = True
                break
            frame = buffer[idx]
            reward = frame["reward"]
            if ret is None:
                ret = torch.zeros_like(reward)
            ret = ret + discount * reward
            if self._as_bool(frame["done"]):
                done_n = True
                break
            discount *= self.gamma

        if ret is None:
            ret = torch.tensor(0.0, dtype=torch.float32)
            done_n = True
        return ret, done_n

    def _normalize_robot_tensors(self, robot_obs, adj):
        if self.num_robots is None and self.robot_obs_dim is None:
            return robot_obs, adj

        cur_n = int(robot_obs.shape[0])
        cur_d = int(robot_obs.shape[1])
        tgt_n = self.num_robots if self.num_robots is not None else cur_n
        tgt_d = self.robot_obs_dim if self.robot_obs_dim is not None else cur_d

        if cur_n == tgt_n and cur_d == tgt_d:
            return robot_obs, adj

        out_obs = torch.zeros((tgt_n, tgt_d), dtype=robot_obs.dtype)
        copy_n = min(cur_n, tgt_n)
        copy_d = min(cur_d, tgt_d)
        out_obs[:copy_n, :copy_d] = robot_obs[:copy_n, :copy_d]

        out_adj = torch.zeros((tgt_n, tgt_n), dtype=adj.dtype)
        copy_adj_n = min(cur_n, tgt_n)
        out_adj[:copy_adj_n, :copy_adj_n] = adj[:copy_adj_n, :copy_adj_n]
        return out_obs, out_adj

    def __iter__(self):
        if wds is None:
            raise RuntimeError("webdataset is not installed.")

        try:
            dataset = wds.WebDataset(
                self.shards,
                shardshuffle=100,
                nodesplitter=getattr(wds, "split_by_node", None),
                workersplitter=getattr(wds, "split_by_worker", None),
            ).decode("pil")
        except TypeError:
            dataset = wds.WebDataset(self.shards, shardshuffle=100).decode("pil")
            if hasattr(dataset, "split_by_node"):
                dataset = dataset.split_by_node()
            if hasattr(dataset, "split_by_worker"):
                dataset = dataset.split_by_worker()

        current_ep = None
        buffer = []

        def flush_buffer():
            extra = 0
            if self.include_nstep_bootstrap:
                extra = self.n_step
            elif self.include_next:
                extra = 1
            min_len = self.clip_len + extra
            if len(buffer) < min_len:
                return
            max_i = len(buffer) - self.clip_len - extra
            start_idxs = list(range(0, max_i + 1, self.clip_stride))
            if len(start_idxs) > 1:
                perm = torch.randperm(len(start_idxs)).tolist()
                start_idxs = [start_idxs[p] for p in perm]

            for i in start_idxs:
                clip = self._apply_done_termination(buffer[i : i + self.clip_len])

                raw_video = [f["image"] for f in clip]
                raw_next_video = None
                if self.include_next:
                    next_clip = self._apply_done_termination(buffer[i + 1 : i + 1 + self.clip_len])
                    raw_next_video = [f["image"] for f in next_clip]
                nstep_clip = None
                if self.include_nstep_bootstrap:
                    nstep_clip = self._apply_done_termination(buffer[i + self.n_step : i + self.n_step + self.clip_len])
                    raw_nstep_video = [f["image"] for f in nstep_clip]

                text = clip[0]["text"]
                inputs = preprocess_vlm_video_inputs(
                    vlm_processor=self.vlm_processor,
                    frames=raw_video,
                    text=text,
                    text_prompt_template=self.text_prompt_template,
                    vlm_max_text_len=self.vlm_max_text_len,
                    vlm_truncation=self.vlm_truncation,
                    vlm_padding=self.vlm_padding,
                    squeeze_batch_dim=True,
                )
                next_inputs = (
                    preprocess_vlm_video_inputs(
                        vlm_processor=self.vlm_processor,
                        frames=raw_next_video,
                        text=text,
                        text_prompt_template=self.text_prompt_template,
                        vlm_max_text_len=self.vlm_max_text_len,
                        vlm_truncation=self.vlm_truncation,
                        vlm_padding=self.vlm_padding,
                        squeeze_batch_dim=True,
                    )
                    if self.include_next
                    else None
                )
                nstep_inputs = (
                    preprocess_vlm_video_inputs(
                        vlm_processor=self.vlm_processor,
                        frames=raw_nstep_video,
                        text=text,
                        text_prompt_template=self.text_prompt_template,
                        vlm_max_text_len=self.vlm_max_text_len,
                        vlm_truncation=self.vlm_truncation,
                        vlm_padding=self.vlm_padding,
                        squeeze_batch_dim=True,
                    )
                    if self.include_nstep_bootstrap
                    else None
                )

                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                adj = torch.stack([f["adj"] for f in clip], dim=0)
                if self.include_next:
                    next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)
                    next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)
                if self.include_nstep_bootstrap:
                    nstep_robot_obs = torch.stack([f["robot_obs"] for f in nstep_clip], dim=0)
                    nstep_adj = torch.stack([f["adj"] for f in nstep_clip], dim=0)

                reward = clip[-1]["reward"]
                done = clip[-1]["done"]
                if self.return_horizon == "trajectory":
                    traj = []
                    for j in range(i + self.clip_len, len(buffer)):
                        traj.append(buffer[j])
                        if self._as_bool(buffer[j]["done"]):
                            break
                    returns = self._discounted_return(traj)
                else:
                    returns = self._discounted_return(clip)

                out = {
                    "inputs": inputs,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "reward": reward.view(1),
                    "returns": returns.view(1),
                    "done": done.view(1),
                    "shard_id": clip[0]["shard_id"],
                }
                if self.include_next:
                    out["next_inputs"] = next_inputs
                    out["next_robot_obs"] = next_robot_obs
                    out["next_adj"] = next_adj
                if self.include_nstep_bootstrap:
                    nstep_returns, nstep_done = self._nstep_discounted_return(buffer, i)
                    out["td_nstep_return"] = nstep_returns.view(1)
                    out["td_nstep_done"] = torch.tensor([float(nstep_done)], dtype=torch.float32)
                    out["td_nstep_inputs"] = nstep_inputs
                    out["td_nstep_robot_obs"] = nstep_robot_obs
                    out["td_nstep_adj"] = nstep_adj
                yield out

        for sample in dataset:
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")
            shard_url = sample.get("__url__", "")
            if isinstance(shard_url, bytes):
                shard_url = shard_url.decode("utf-8", errors="ignore")
            shard_id = os.path.basename(str(shard_url)) if shard_url else "unknown_shard"
            ep_id = _extract_episode_id(key)

            if current_ep is None:
                current_ep = ep_id
            if ep_id != current_ep:
                yield from flush_buffer()
                buffer = []
                current_ep = ep_id

            image = sample["image.png"]
            robot_src = _as_numpy(sample[f"{self.robot_source}.npy"])
            if hasattr(robot_src, "numpy"):
                robot_src = robot_src.numpy()
            robot_obs = torch.tensor(robot_src, dtype=torch.float32)

            num_nodes = robot_obs.shape[0]
            adj = _edge_index_to_adj(_as_numpy(sample["edge_index.npy"]), num_nodes)
            robot_obs, adj = self._normalize_robot_tensors(robot_obs, adj)

            if self.text_mode == "raw":
                text = self.text_prompt_template
            else:
                text = _as_numpy(sample["text_emb.npy"])
                if hasattr(text, "numpy"):
                    text = text.numpy()
                text = torch.tensor(text, dtype=torch.float32)

            reward = _reward_from_frame(sample, self.reward_reduce)
            done = _done_from_frame(sample, self.done_reduce)

            buffer.append(
                {
                    "image": image,
                    "robot_obs": robot_obs,
                    "adj": adj,
                    "text": text,
                    "reward": reward,
                    "done": done,
                    "shard_id": shard_id,
                }
            )

        yield from flush_buffer()


def _collate_sequence_batch(batch):
    def _stack_inputs(items):
        out = {}
        keys = items[0].keys()
        for k in keys:
            vals = [d[k] for d in items]
            if torch.is_tensor(vals[0]):
                out[k] = torch.stack(vals, dim=0)
            else:
                out[k] = vals
        return out

    out = {
        "inputs": _stack_inputs([b["inputs"] for b in batch]),
        "robot_obs": torch.stack([b["robot_obs"] for b in batch], dim=0),
        "adj": torch.stack([b["adj"] for b in batch], dim=0),
        "reward": torch.stack([b["reward"] for b in batch], dim=0).view(-1),
        "done": torch.stack([b["done"] for b in batch], dim=0).view(-1),
    }
    if "next_inputs" in batch[0]:
        out["next_inputs"] = _stack_inputs([b["next_inputs"] for b in batch])
        out["next_robot_obs"] = torch.stack([b["next_robot_obs"] for b in batch], dim=0)
        out["next_adj"] = torch.stack([b["next_adj"] for b in batch], dim=0)
    if "returns" in batch[0]:
        out["returns"] = torch.stack([b["returns"] for b in batch], dim=0).view(-1)
    if "td_nstep_return" in batch[0]:
        out["td_nstep_return"] = torch.stack([b["td_nstep_return"] for b in batch], dim=0).view(-1)
        out["td_nstep_done"] = torch.stack([b["td_nstep_done"] for b in batch], dim=0).view(-1)
        out["td_nstep_inputs"] = _stack_inputs([b["td_nstep_inputs"] for b in batch])
        out["td_nstep_robot_obs"] = torch.stack([b["td_nstep_robot_obs"] for b in batch], dim=0)
        out["td_nstep_adj"] = torch.stack([b["td_nstep_adj"] for b in batch], dim=0)
    return out


class UniqueShardBatchDataset(IterableDataset):
    def __init__(self, base_dataset, batch_size, max_queue_per_shard=16, drop_last=True):
        self.base_dataset = base_dataset
        self.batch_size = int(batch_size)
        self.max_queue_per_shard = int(max_queue_per_shard)
        self.drop_last = bool(drop_last)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.max_queue_per_shard <= 0:
            raise ValueError("max_queue_per_shard must be > 0")

    def __iter__(self):
        per_shard = defaultdict(deque)

        for sample in self.base_dataset:
            shard_id = sample.get("shard_id", "unknown_shard")
            q = per_shard[shard_id]
            if len(q) >= self.max_queue_per_shard:
                q.popleft()
            q.append(sample)

            while len(per_shard) >= self.batch_size:
                shard_ids = list(per_shard.keys())
                if len(shard_ids) > self.batch_size:
                    perm = torch.randperm(len(shard_ids))[: self.batch_size].tolist()
                    shard_ids = [shard_ids[i] for i in perm]
                else:
                    shard_ids = shard_ids[: self.batch_size]

                batch = []
                empty_keys = []
                for sid in shard_ids:
                    sq = per_shard[sid]
                    batch.append(sq.popleft())
                    if len(sq) == 0:
                        empty_keys.append(sid)
                for sid in empty_keys:
                    del per_shard[sid]
                yield batch

        if not self.drop_last:
            leftovers = []
            for q in per_shard.values():
                while len(q) > 0:
                    leftovers.append(q.popleft())
            if leftovers:
                yield leftovers


def _collate_prebatched_sequence_batch(batch):
    return _collate_sequence_batch(batch)


def webdataset_loader(args, shards, batch_size, num_workers):
    shards = resolve_shards_spec(shards)
    vlm_processor = None
    if args.preprocess_in_loader:
        from transformers import AutoProcessor

        vlm_processor = AutoProcessor.from_pretrained(args.vl_model_name)
        tokenizer = getattr(vlm_processor, "tokenizer", None)
        if tokenizer is not None and "<obs>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<obs>"]})

    dataset = SequenceWebDataset(
        shards=shards,
        clip_len=args.clip_len,
        clip_stride=args.clip_stride,
        text_mode=args.text_mode,
        robot_source=args.robot_source,
        reward_reduce=args.reward_reduce,
        done_reduce=args.done_reduce,
        vlm_processor=vlm_processor,
        text_prompt_template=args.text_prompt_template,
        include_next=(args.loss_type != "contrastive" and args.return_mode == "td"),
        include_nstep_bootstrap=(args.loss_type != "contrastive" and args.return_mode == "nstep"),
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=(args.vl_backend != "llava_video"),
        vlm_padding=("longest" if args.vl_backend == "llava_video" else "max_length"),
        gamma=getattr(args, "gamma", 0.99),
        return_horizon=getattr(args, "return_horizon", "clip"),
        n_step=getattr(args, "n_step", 1),
        num_robots=getattr(args, "num_robots", None),
        robot_obs_dim=getattr(args, "robot_obs_dim", None),
    )
    if getattr(args, "shard_aware_batching", False):
        dataset = UniqueShardBatchDataset(
            base_dataset=dataset,
            batch_size=batch_size,
            max_queue_per_shard=getattr(args, "shard_batch_max_queue_per_shard", 16),
            drop_last=True,
        )
        return DataLoader(dataset, batch_size=None, num_workers=num_workers, collate_fn=_collate_prebatched_sequence_batch)

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_sequence_batch)
