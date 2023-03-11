"""Running MUSAE."""

from musae import MUSAE
from utils import tab_printer
from parser import parameter_parser
from train import train
from torch_geometric.datasets import WikipediaNetwork

class Args:
    model_type: str = 'GAT'
    num_layers: int = 2
    heads: int = 2
    batch_size: int = 32
    hidden_dim: int = 32
    dropout: float = 0.5
    epochs: int = 500
    opt: str = 'adam'
    opt_scheduler: str = 'none' 
    opt_restart: int = 0
    weight_decay: float = 5e-3
    lr: float = 0.01


def main(args):
    """
    Multi-scale attributed node embedding machine calling wrapper.
    :param args: Arguments object parsed up.
    """
    # model = MUSAE(args)
    # model.do_sampling()
    # model.learn_embedding()
    # model.save_embedding()
    # model.save_logs()
    # model.graph
    args = Args()
    dataset = WikipediaNetwork('./data', 'chameleon')
    test_accs, losses, best_model, best_acc, test_loader = train(dataset, args) 
    print("Maximum test set accuracy: {0}".format(max(test_accs)))
    print("Minimum loss: {0}".format(min(losses)))


if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    main(args)
