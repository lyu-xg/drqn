
import os
import sys
import signal
import webbrowser
from multiprocessing import Process
import subprocess

HTML_OUT = 'index.html'

def run_tb(dir_name, port):
    os.chdir(dir_name)
    subprocess.call(['tensorboard', '--port', str(port), '--logdir', '.'])

def sig_han(sig, frames):
    if os.path.isfile(HTML_OUT):
        os.remove(HTML_OUT)
        print('\nokay, bye')
    raise SystemExit

def main():
    os.chdir(sys.path[0] + '/log')
    log_dirs = os.listdir()
    fp = open(HTML_OUT, 'w')
    fp.write('<html><body>')
    
    for port_suffix, log_dir in enumerate(log_dirs):
        port = 6000 + port_suffix
        Process(target=run_tb, args=(log_dir, port)).start()
        fp.write('\n\t<a href="http://localhost:{}">{}</a></br>'.format(port, log_dir))
    fp.write('\n</body></html>')
    webbrowser.open_new_tab(HTML_OUT)
    fp.close()
    

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_han)
    main()
