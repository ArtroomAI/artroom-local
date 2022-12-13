import sys
import subprocess
import pkg_resources

def install_ltt():
    required = {'light-the-torch'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed
    
    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install','--user', '--no-warn-script-location',*missing])


def install_cuda():
    from light_the_torch._cli import main as ltt
    required = {'torch'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        print('Missing packages: {}'.format(', '.join(missing)))
        print('Installing missing packages...')
        ltt(['install', '--user', *missing])
        import torch
        if torch.cuda.is_available():
            print('CUDA is available.')
        else:
            print('Unsupported CUDA version or GPU.')
    else:
        print('Torch is installed.')
        import torch
        print('Torch version: {}'.format(torch.__version__))
        if torch.cuda.is_available():
            print('CUDA is available.')
            print('CUDA version: {}'.format(torch.version.cuda))
        else:
            print('CUDA is not available.')
            print('Reinstalling Torch with CUDA support...')
            ltt(['install','--user', '--upgrade', '--force-reinstall',  'torch'])
