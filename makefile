clean:
	rm -rf ckpts log __pycache__ ipynb_checkpoints; mkdir ckpts log

back:
	cp -r ckpts backup; cp -r log backup
