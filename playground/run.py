import os
from multiprocessing import Process

# still work in progress....!

def go_run(cmd):
    os.system(cmd)

def main(**kwarg):
    for h_size in (512, 256):
        for env in ('SpaceInvaders', 'Frostbite', 'MsPacman'):
            Process(
                target=go_run,
                args=('python drqn.py -d {} -e {}'.format(h_size, env),
            )).start()

if __name__ == '__main__':
    main()