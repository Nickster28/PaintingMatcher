import argparse
from SimpleResizeModel import SimpleResizeModel
from HistogramResizeModel import HistogramResizeModel
from GramResizeModel import GramResizeModel
from GramHistoResizeModel import GramHistoResizeModel
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='SimpleResizeModel', type=str)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dataset_size', default=-1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--log_dir', default='log', type=str)

args = parser.parse_args()

if args.model == 'SimpleResizeModel':
	model = SimpleResizeModel(args.batch_size, args.dataset_size,
		args.num_workers, args.num_epochs, args.learning_rate,
		args.dropout_keep_prob, args.weight_decay, args.log_dir)
	model.train()
elif args.model == 'HistogramResizeModel':
	model = HistogramResizeModel(args.batch_size, args.dataset_size,
		args.num_workers, args.num_epochs, args.learning_rate,
		args.dropout_keep_prob, args.weight_decay, args.log_dir)
	model.train()
elif args.model == 'GramResizeModel':
	model = GramResizeModel(args.batch_size, args.dataset_size,
		args.num_workers, args.num_epochs, args.learning_rate,
		args.dropout_keep_prob, args.weight_decay, args.log_dir)
	model.train()
elif args.model == 'GramHistoResizeModel':
	model = GramHistoResizeModel(args.batch_size, args.dataset_size,
		args.num_workers, args.num_epochs, args.learning_rate,
		args.dropout_keep_prob, args.weight_decay, args.log_dir)
	model.train()
else:
	print("Error: unknown model " + args.model)