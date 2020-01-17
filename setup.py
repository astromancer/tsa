
import site
import inspect
import subprocess
from pathlib import Path


def link_into_path():
    """
    Simple install strategy that links the repo's man directory into system python path.
    """
    here = inspect.getfile(inspect.currentframe())
    here = Path(here)
    src = here.parent.resolve()
    pkg_name = src.name
    src = str(src)
    dest = site.getsitepackages()[0]
    dest = str(Path(dest) / pkg_name)

    #print(src, dest)
    subprocess.call(['ln','-s', src, dest])
    # ln -s path/to/repo/eeg `python3 -m site --user-site`/eeg


link_into_path()