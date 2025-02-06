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

# torch.manual_seed(1337)

class Trainer:
    def __init__(self, arg):

        cfg_path = arg.config
        self.tag = arg.tag
        self.config_path = cfg_path
        self.cfg = self.load_config(cfg_path)
        self.update_config()

        self.build_model()
        self.build_optimizer()
        self.dataloader = DataLoader(self.cfg.datapath)
        if self.base_model is not None:
            self.load_config()

    def update_config(self):
        self.optimizer = self.cfg.optimizer
        self.dataloader = self.cfg.dataloader
        self.batch_size = self.cfg.batch_size
        self.block_size = self.cfg.block_size
        self.device = self.cfg.get("device", "cpu")
        assert self.device in ("cuda", "cpu")
        self.base_model = self.cfg.get("base_model", None)

    def build_model(self):
        self.model = BigramLanguageModel(self.cfg)

    def load_config(config_file_path):
        # Load the configuration file as a module
        spec = importlib.util.spec_from_file_location("config_module", config_file_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules["config_module"] = config_module
        spec.loader.exec_module(config_module)
        
        # Return the module for accessing the configuration
        return config_module

    def save_model(self, epoch):
        model_state_dict = self.model.state_dict()
        data_to_save = {"model_state_dict" : model_state_dict,
                        "config" : self.cfg}
        workspace_dir = os.dirname(os.dirname(__file__))
        save_path = os.path.join((workspace_dir, "ckpts/{}.safetensors"))
        base_file_name = self.config_path.split("/")[-1]
        filename = base_file_name.split()[0]
        filename = "_".join(filename, self.tag, str(epoch))
        filename = filename + ".safetensors"
        st.save_file(data_to_save, os.path.abspath(""))

    def load_model(self, path):
        data = st.load_file(path)
        model_state_dict = data["model_state_dict"]
        config = data["config"]

        self.model.load_state_dict(model_state_dict)




    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                X, Y = self.data_loader.get_batch(split, self.batch_size, self.block_size, self.device)
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

            xb, yb = self.data_loader.get_batch("train", self.batch_size, self.block_size, self.device)
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--tag")

    return parser.parse()


if __name__ == "__main__":
    arg = parseargs()
    trainer = Trainer(arg)
    trainer.train()