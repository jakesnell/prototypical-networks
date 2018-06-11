import argparse

from eval import main

parser = argparse.ArgumentParser(description='Evaluate few-shot prototypical networks')

default_model_path = 'results/best_model.pt'
parser.add_argument('--model.model_path', type=str, default=default_model_path, metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: {:s})".format(default_model_path))

parser.add_argument('--data.test_way', type=int, default=0, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as model's data.test_way (default: 0)")
parser.add_argument('--data.test_shot', type=int, default=0, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as model's data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=0, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as model's data.query (default: 0)")
parser.add_argument('--data.test_episodes', type=int, default=1000, metavar='NTEST',
                    help="number of test episodes per epoch (default: 1000)")

args = vars(parser.parse_args())

main(args)
