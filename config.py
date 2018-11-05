import argparse
from drqn import trian as drqn_train
from dqn import train as dqn_train

RUNS = [
    (dqn_train, {'env_name': 'Frostbite'}),
    (dqn_train, {'env_name': 'Asteroids'}),
    (dqn_train, {'env_name': 'Pong'})

]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--run_config', action='store', type=int, default=-1)
    train, config = RUNS[parser.run_config]
    train(**config)

if __name__ == '__main__':
    main()