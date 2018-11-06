import argparse
from drqn import train as drqn_train
from dqn import train as dqn_train

RUNS = [
    # (drqn_train, {'num_quant': 128, 'env_name': 'Asteroids'}),

    # (drqn_train, {'env_name': 'Pong'}),
    # (drqn_train, {'env_name': 'Frostbite'}),
    # (drqn_train, {'env_name': 'MsPacman'}),
    # (drqn_train, {'env_name': 'Asteroids'}),

    (dqn_train, {'env_name': 'Pong'}),
    (dqn_train, {'env_name': 'Frostbite'}),
    (dqn_train, {'env_name': 'MsPacman'}),
    (dqn_train, {'env_name': 'Asteroids'}),

    (drqn_train, {'num_quant': 128, 'env_name': 'flicker@.5:Pong'}),
    (drqn_train, {'num_quant': 128, 'env_name': 'flicker@.5:Frostbite'}),
    (drqn_train, {'num_quant': 128, 'env_name': 'flicker@.5:MsPacman'}),
    (drqn_train, {'num_quant': 128, 'env_name': 'flicker@.5:Asteroids'}),

    # (drqn_train, {'env_name': 'flicker@.5:Pong'}),
    # (drqn_train, {'env_name': 'flicker@.5:Frostbite'}),
    # (drqn_train, {'env_name': 'flicker@.5:MsPacman'}),
    # (drqn_train, {'env_name': 'flicker@.5:Asteroids'}),

    (dqn_train, {'env_name': 'flicker@.5:Pong'}),
    (dqn_train, {'env_name': 'flicker@.5:Frostbite'}),
    (dqn_train, {'env_name': 'flicker@.5:MsPacman'}),
    (dqn_train, {'env_name': 'flicker@.5:Asteroids'}),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--run_config', action='store', type=int, default=-1)
    train, config = RUNS[parser.parse_args().run_config]
    train(**config)

if __name__ == '__main__':
    main()