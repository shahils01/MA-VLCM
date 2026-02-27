import io

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
            min_len = self.clip_len + (1 if self.include_next else 0)
            if len(buffer) < min_len:
                return
            max_i = len(buffer) - self.clip_len - (1 if self.include_next else 0)
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

                if self.vlm_processor is None:
                    raise RuntimeError("Dataloader processor not set.")

                def _proc(frames, text):
                    if not isinstance(text, str):
                        text = self.text_prompt_template
                    tokenizer = getattr(self.vlm_processor, "tokenizer", None)
                    if tokenizer is not None:
                        vocab = tokenizer.get_vocab()
                        if "<video>" in vocab and "<video>" not in text and "<image>" not in text:
                            text = f"<video>\\n{text}"
                        if "<obs>" in vocab and "<obs>" not in text:
                            if "<video>" in text:
                                text = text.replace("<video>\\n", "<video><obs>\\n", 1)
                            else:
                                text = f"<obs>\\n{text}"
                    try:
                        max_len = self.vlm_max_text_len if self.vlm_truncation else None
                        inputs = self.vlm_processor(
                            text=text,
                            videos=frames,
                            return_tensors="pt",
                            padding=self.vlm_padding,
                            truncation=self.vlm_truncation,
                            max_length=max_len,
                        )
                    except TypeError:
                        inputs = self.vlm_processor(images=frames, return_tensors="pt")

                    packed = {}
                    for k, v in dict(inputs).items():
                        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
                            v = v.squeeze(0)
                        packed[k] = v
                    return packed

                text = clip[0]["text"]
                inputs = _proc(raw_video, text)
                next_inputs = _proc(raw_next_video, text) if self.include_next else None

                robot_obs = torch.stack([f["robot_obs"] for f in clip], dim=0)
                adj = torch.stack([f["adj"] for f in clip], dim=0)
                if self.include_next:
                    next_robot_obs = torch.stack([f["robot_obs"] for f in next_clip], dim=0)
                    next_adj = torch.stack([f["adj"] for f in next_clip], dim=0)

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
                }
                if self.include_next:
                    out["next_inputs"] = next_inputs
                    out["next_robot_obs"] = next_robot_obs
                    out["next_adj"] = next_adj
                yield out

        for sample in dataset:
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8", errors="ignore")
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
    return out


def webdataset_loader(args, shards, batch_size, num_workers):
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
        vlm_max_text_len=args.vl_max_text_len,
        vlm_truncation=(args.vl_backend != "llava_video"),
        vlm_padding=("longest" if args.vl_backend == "llava_video" else "max_length"),
        gamma=getattr(args, "gamma", 0.99),
        return_horizon=getattr(args, "return_horizon", "clip"),
    )

    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=_collate_sequence_batch)
