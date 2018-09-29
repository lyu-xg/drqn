import os
from multiprocessing import Process
import subprocess


def run_tb(dir_name):
    os.chdir(dir_name)
    trace = [param.split('=')
             for param in dir_name.split(',')
             if param.startswith('stack')][0][1]
    port = ('600' if int(trace) < 10 else '60') + trace
    print('starting TensorBoard for trace_length={} at port={}'.format(trace, port))
    subprocess.call(['tensorboard', '--port', port, '--logdir', '.'])


def main():
    os.chdir(os.getenv("HOME") + '/drqn/log')
    log_dirs = os.listdir()
    for log_dir in log_dirs:
        Process(target=run_tb, args=(log_dir,)).start()


if __name__ == '__main__':
    main()
