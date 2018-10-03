
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
    mod, trace = identity['mod'], identity['stack']
    port = PORT_PREFIX[mod] + ('0' if len(trace) < 2 else '') + trace
    print('starting TensorBoard for model={}, {}_length={} at http://localhost:{}'.format(
        mod, 'trace' if mod!='dqn' else 'stack', trace, port))
    subprocess.check_output(['tensorboard', '--port', port, '--logdir', '.'])


def main():
    os.chdir(os.getenv("HOME") + '/drqn/log')
    log_dirs = os.listdir()
    for log_dir in log_dirs:
        Process(target=run_tb, args=(log_dir,)).start()


if __name__ == '__main__':
    main()
