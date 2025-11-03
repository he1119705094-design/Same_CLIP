import subprocess

n1 = 'sdpure_real.csv'
n2 = 'sdpure_fake.csv'
wdir = './weights/'
models = 'ours-sync'
commands = [
    f"python main.py --in_csv data/{n1} --out_csv output/{n1} --device 'cuda:4' --weights_dir {wdir} --models {models}",
    f"python main.py --in_csv data/{n2} --out_csv output/{n2} --device 'cuda:5' --weights_dir {wdir} --models {models}",
]

processes = [subprocess.Popen(command, shell=True) for command in commands]

for process in processes:
    process.wait()

