
import os
from multiprocessing import Process
import subprocess


PORT_PREFIX = {
    'dqn': '60',
    'drqn': '70',
    'adrqn': '80'
}

def run_tb(dir_name):
    os.chdir(dir_name)
    identity = dict(param.split('=') for param in dir_name.split(','))
    print(identity)
    mod, trace = identity['mod'], identity['stack']
    print(PORT_PREFIX[mod])
    port = PORT_PREFIX[mod] + ('0' if len(trace) < 2 else '') + trace
    print('starting TensorBoard for trace_length={} at port={}'.format(trace, port))
    # subprocess.call(['tensorboard', '--port', port, '--logdir', '.'])


def main():
    os.chdir(os.getenv("HOME") + '/drqn/log')
    log_dirs = os.listdir()
    for log_dir in log_dirs:
        Process(target=run_tb, args=(log_dir,)).start()


if __name__ == '__main__':
    main()
