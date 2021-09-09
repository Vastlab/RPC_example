import argparse
import numpy as np
import os
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote

import time


torch.manual_seed(2)



def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

  
  
class net_class(torch.nn.Module):
  def __init__(self):
    super(net_class, self).__init__()
    self.affine1 = torch.nn.Linear(64, 16)
    self.affine2 = torch.nn.Linear(16, 4)

  def forward(self, x):
    x = self.affine1(x)
    feature = torch.nn.functional.relu(x)
    Logit = self.affine2(x)
    predicted_class = torch.nn.functional.softmax(Logit, dim=1)
    return feature, Logit, predicted_class



class Feature_Extractor(object):
  def __init__(self):
    rank = 1
    self.id = rpc.get_worker_info().id
    self.gpu = rank - 1
    self.net = net_class().cuda(self.gpu)
  
  def run(self, agent_rref, input_tensor):
    """
    Arguments:
      agent_rref (RRef): an RRef referencing the agent object.
    """
    x = input_tensor.cuda(self.gpu)
    # extract feature from images
    feature, Logit, predicted_class = self.net(x)
    
    feature = feature.cpu()
    Logit = Logit.cpu()
    predicted_class = predicted_class.cpu()
    

    # report the feature to the agent
    _remote_method(Agent.report_feature, agent_rref, self.id, feature, Logit, predicted_class)



class EVM_RPC(object):
  def __init__(self):
    rank = 2
    self.id = rpc.get_worker_info().id
    self.gpu = rank - 1

  def predict(self,FV):
    # fake function
    # to be change with real EVM
    return torch.tanh(FV**2)


  def run(self, agent_rref, feature_tensor):
    """
    Arguments:
      agent_rref (RRef): an RRef referencing the agent object.
    """
    
    x = feature_tensor.cuda(self.gpu)

    # predict the parbability of novety
    probability = self.predict(x)
    
    probability = probability.cpu()

    # report the probability to the agent
    _remote_method(Agent.report_probability, agent_rref, self.id, probability)



class Agent(object):
  
  def __init__(self):
    fe_info = rpc.get_worker_info("feature_extractor")
    evm_info = rpc.get_worker_info("evm")
    self.fe_rref = remote(fe_info, Feature_Extractor)
    self.evm_rref = remote(evm_info, EVM_RPC)
    self.agent_rref = RRef(self)


  def report_feature(self, fe_id, feature, Logit, predicted_class):
    """
    feature extractor call this function to report feature tensor.
    """
    self.feature = feature
    self.Logit = Logit
    self.predicted_class = predicted_class


  def report_probability(self, evm_id, probability):
    """
    EVM call this function to report probability.
    """
    self.prediction = probability


  def run_feature_extractor(self, input_tensor):
    """
    The agent will tell feature extractr run.
    """
    # make async RPC to kick off an episode on all observers
    fut = rpc_async( self.fe_rref.owner(),  _call_method,
                     args=(Feature_Extractor.run, self.fe_rref, 
                           self.agent_rref, input_tensor)  )
    fut.wait()
    return self.feature


  def run_classification(self, feature):
    """
    The agent will tell feature extractr run.
    """
    # make async RPC to kick off an episode on all observers
    fut = rpc_async( self.evm_rref.owner(),  _call_method,
                     args=(EVM_RPC.run, self.evm_rref, self.agent_rref, feature)  )
    fut.wait()
    return self.prediction



def worker(rank, world_size):
  """
  This is the entry point for all processes. 
  The rank 0 is the agent. 
  The rank 1 is the feature extractor.
  The rank 2 is the EVM. 
  """
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '29500'
  if rank == 0:
    # rank 0 is the agent, no gpu
    rpc.init_rpc("agent", rank=0, world_size=world_size)

    agent = Agent()
    
    # TBD: change the for loop from range(5) to data loader
    for batch_id in range(5):
      t1 = time.time()
      print("\nbatch = ", batch_id + 1)
      input_tensor = torch.rand(20,64) # random input tensor (from data loader)
      feature_tensor = agent.run_feature_extractor(input_tensor)
      output_tensor = agent.run_classification(feature_tensor)
      print("output_tensor.shape = ", output_tensor.shape)
      t2 = time.time()
      print("batch time = ", t2 - t1)

  elif rank == 1:
    # rank 0 is the feature extractor, gpu = 0
    rpc.init_rpc("feature_extractor", rank=1, world_size=world_size)
    # feature extractor passively waiting for instructions from agents
  elif rank == 2:
    # rank 2 is the EVM, gpu = 1
    rpc.init_rpc("evm", rank=2, world_size=world_size)
    # EVM passively waiting for instructions from agents
  else:
    raise ValueError()
  rpc.shutdown()



if __name__ == '__main__':
  number_of_gpu = 2
  world_size = 1 + number_of_gpu
  mp.spawn( worker, args=(world_size, ), nprocs=world_size, join=True  )
