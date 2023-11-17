from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
from ordinalentropy import ordinalentropy
from losses import Depth_Loss


class OER(nn.Module):
    """OER adapts a model by entropy minimization during testing.

    Once oered, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"  # if not steps >=0, then trigger error
        self.episodic = episodic



        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory

        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, weak=None, strong=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            print(weak)
            if weak!=None:
                print("sdjasjdowqeqwe")
                outputs = forward_and_adapt_fixmatch(x, weak, strong, self.model, self.optimizer)
            else:
                outputs = forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


# @torch.jit.script
# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    optimizer.zero_grad()
    outputs, features = model(x)
    outputs = outputs.detach()                  # detach the target before computing the loss  https://stackoverflow.com/questions/72590591/the-derivative-for-target-is-not-implemented
    # features = features.detach()
    # adapt
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    loss = ordinalentropy(features, outputs)
    # loss = torch.nn.L1Loss()
    # loss = loss(features, outputs)

    # loss_func = Depth_Loss(1, 1, 1, maxDepth=80)
    # loss = loss_func(features, outputs)
    print("loss: ", loss)
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    loss.backward()
    optimizer.step()
    # print("{:.3f}MB allocated".format(torch.cuda.memory_allocated() / 1024 ** 2))
    return outputs

def forward_and_adapt_fixmatch(image, weak, strong, model, optimizer):
    """Forward and adapt model on batch of data.
       take gradients, and update params.
    """
    # forward
    predictions_weaks, _ = model(weak)
    predictions_strongs, _ = model(strong)
    # predictions_weaks = predictions_weaks.detach()   # detach the target before computing the loss  https://stackoverflow.com/questions/72590591/the-derivative-for-target-is-not-implemented

    # adapt
    loss = torch.nn.L1Loss()
    loss = loss(predictions_strongs, predictions_weaks)
    print("loss: ", loss)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    prediction, _ = model(image)
    return prediction


#
# @torch.enable_grad()  # ensure grads in possible no grad context for testing
# def forward_and_adapt(x, model, optimizer):
#     """Forward and adapt model on batch of data.
#
#     Measure entropy of the model prediction, take gradients, and update params.
#     """
#     # forward
#     outputs = model(x)
#     # adapt
#     loss = softmax_entropy(outputs).mean(0)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
#     return outputs

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

