
import os
import sys
import signal
import webbrowser
from multiprocessing import Process
import subprocess

HTML_OUT = '../tensorboard/index.html'
TEMPLATE = '../tensorboard/index.template'
LINE_TEMPLATE = '\t\t\t\t<li class="collection-item" id="{}">{}</li>'


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
    lines = []

    for port_suffix, log_dir in enumerate(log_dirs):
        port = 6001 + port_suffix
        Process(target=run_tb, args=(log_dir, port)).start()
        lines.append(LINE_TEMPLATE.format(port, log_dir))
        # fp.write('\n\t<a href="http://localhost:{}">{}</a></br>'.format(port, log_dir))
    # fp.write('\n</body></html>')
    with open(TEMPLATE) as t, open(HTML_OUT, 'w') as o:
        o.write(t.read().format('\n'.join(lines)))
    # webbrowser.open_new_tab(os.path.abspath(HTML_OUT))
    # print(os.path.abspath(HTML_OUT))
    webbrowser.open_new_tab('file://' + os.path.abspath(HTML_OUT))
    

if __name__ == '__main__':
    signal.signal(signal.SIGINT, sig_han)
    main()
