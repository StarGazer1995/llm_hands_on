import torch
import torch.nn as nn
from torch.nn import functional as F
from safetensors import torch as st
from data_loader import DataLoader
from gpt_model import BigramLanguageModel
import argparse
import importlib.util
import sys
import os
import inspect

# torch.manual_seed(1337)

class Trainer:
    def __init__(self, arg):

        cfg_path = arg.config
        self.tag = arg.tag
        self.config_path = cfg_path
        self.cfg = self.load_config(cfg_path)
        self.model_cfg = self.cfg.model
        self.update_config()

        self.build_model()
        self.build_optimizer()
        self.data_loader = DataLoader(self.dataset, self.dataset, device=self.model_cfg["device"], mode="train", batch_size=self.batch_size, block_size=self.block_size)

    def update_config(self):
        self.batch_size = self.model_cfg.get("batch_size", None)
        self.block_size = self.model_cfg.get("block_size", None)
        self.device = self.model_cfg.get("device", "cpu")
        assert self.device in ("cuda", "cpu")
        self.base_model = self.model_cfg.get("base_model", None)
        self.max_iters = self.model_cfg.get("max_iters", None)
        self.eval_interval = self.model_cfg.get("eval_interval", None)
        self.eval_iters = self.model_cfg.get("eval_iters", None)
        self.dataset = self.model_cfg.get("datapath", None)

    def build_model(self):
        self.model = BigramLanguageModel(self.model_cfg)
        self.model = self.model.to(self.model_cfg["device"])

    def load_config(self, config_file_path):
        # Load the configuration file as a module
        spec = importlib.util.spec_from_file_location("config_module", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)
        # Return the module for accessing the configuration
        return config_module

    def save_model(self, epoch):
        workspace_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        base_file_name = self.config_path.split("/")[-1]
        filename = base_file_name.split(".")[0]
        save_dir = os.path.join(workspace_dir, "ckpts", filename)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except:
            raise FileNotFoundError
        
        if self.tag is not None:
            filename = f"epoch_{str(epoch)}_{self.tag}"
        else:
            filename = f"epoch_{str(epoch)}"
        source = inspect.getsource(self.cfg)
        filename = os.path.join(save_dir,filename)

        torch.save(self.model, filename+".pth")
        with open(f"{filename}.py", "w") as f:
            f.write(source)

    def load_model(self, path):
        data = st.load_file(path)
        model_state_dict = data["model_state_dict"]
        config = data["config"]

        self.model.load_state_dict(model_state_dict)

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_cfg.get("learning_rate", None))

    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.data_loader.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            xb, yb = self.data_loader.get_batch("train")
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.save_model("last")

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--tag")

    return parser.parse_args()


if __name__ == "__main__":
    arg = parseargs()
    trainer = Trainer(arg)
    trainer.train()