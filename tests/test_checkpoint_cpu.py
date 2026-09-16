"""Small CPU model only, not the final Transformer or real data-loader integration."""
import unittest
import random
try:
    import torch
    import numpy as np
    from freshstart.checkpoint_ref import capture,restore,fingerprint
    from freshstart.core import ContractError
    AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'Torch + NumPy not installed: reference replay not executed')
class CpuReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);random.seed(19);np.random.seed(19);torch.manual_seed(19)
        self.model=torch.nn.Linear(3,2);self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001)
        self.extra={'root_id':4101,'stage':'T3','global_step':2,'loader_state':{'cursor':0},
                    'accumulation_step':0,'config_sha256':'a'*64,'tokenizer_sha256':'b'*64}
    def step(self):
        x=torch.randn(4,3)*(random.random()+float(np.random.random()))
        self.optimizer.zero_grad();loss=self.model(x).square().mean();loss.backward();self.optimizer.step()
        return loss.item()
    def test_model_optimizer_rng_replay(self):
        self.step();self.step();s=capture(self.model,self.optimizer,self.extra)
        losses=[self.step() for _ in range(10)]
        expected=fingerprint({'m':self.model.state_dict(),'o':self.optimizer.state_dict()})
        extra=restore(s,self.model,self.optimizer)
        self.assertEqual(extra,self.extra);self.assertEqual(losses,[self.step() for _ in range(10)])
        self.assertEqual(expected,fingerprint({'m':self.model.state_dict(),'o':self.optimizer.state_dict()}))
    def test_weights_only_rejected(self):
        s=capture(self.model,self.optimizer,self.extra)
        with self.assertRaises(ContractError):restore(s,self.model,None)
    def test_accumulation_midpoint_rejected(self):
        self.extra['accumulation_step']=1
        with self.assertRaises(ContractError):capture(self.model,self.optimizer,self.extra)
    def test_generator_state_required(self):
        g=torch.Generator().manual_seed(7);s=capture(self.model,self.optimizer,self.extra,generators={'loader':g})
        with self.assertRaises(ContractError):restore(s,self.model,self.optimizer,generators={})
    def test_distinct_initialization_fingerprints(self):
        a=fingerprint(self.model.state_dict());torch.manual_seed(20);b=fingerprint(torch.nn.Linear(3,2).state_dict());self.assertNotEqual(a,b)
    def test_state_capture_is_deep(self):
        self.step();s=capture(self.model,self.optimizer,self.extra);before=fingerprint(s['model']);self.step();self.assertEqual(before,fingerprint(s['model']))

if __name__=='__main__':unittest.main()
