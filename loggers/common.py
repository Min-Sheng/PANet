import torch
from torch.utils.tensorboard import SummaryWriter


class BaseLogger:
    """The base class for all loggers.
    Args:
        log_dir (str): The saved directory.
        net (nn.Module): The network architecture.
        dummy_input (torch.Tensor): The dummy input for plotting the network architecture.
    """
    def __init__(self, log_dir, net, dummy_input):
        """
        # TODO: Plot the network architecture.
        # There are some errors: ONNX runtime errors.
        with SummaryWriter(log_dir) as w:
            w.add_graph(net, dummy_input)
        """
        self.writer = SummaryWriter(log_dir)

    def write(self, iter_, log, sample_batched, query_pred, support_pred=None):
        """Plot the network architecture and the visualization results.
        Args:
            iter_ (int): The number of trained iteration.
            log (dict): The log information.
            sample_batched (dict): The sample batch.
            query_pred (torch.Tensor): The query prediction.
            support_pred (torch.Tensor): The support prediction.
        """
        self._add_scalars(iter_, log)
        self._add_images(iter_, sample_batched, query_pred, support_pred)

    def close(self):
        """Close the writer.
        """
        self.writer.close()

    def _add_scalars(self, iter_, log):
        """Plot the training curves.
        Args:
            iter_ (int): The number of trained iteration.
            log (dict): The log information.
        """
        for key in log:
            self.writer.add_scalars(key, {'train': log[key]}, iter_)

    def _add_images(self, iter_, sample_batched, query_pred):
        """Plot the visualization results.
        Args:
            iter_ (int): The number of trained iteration.
            sample_batched (dict): The training batch.
            query_pred (sequence of torch.Tensor): The query prediction.
        """
        raise NotImplementedError
