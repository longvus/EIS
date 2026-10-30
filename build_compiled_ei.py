"""Build the optional compiled EI kernel; compiled_ei.py checks the manifest this script writes and falls back to
NumPy when no kernel is built. Uses $CC (default cc); macOS also needs -undefined dynamic_lookup.

    python build_compiled_ei.py
"""
import hashlib
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy
import scipy

FLAGS = ['-O2', '-shared', '-fPIC', '-fno-fast-math', '-ffp-contract=off', '-fno-builtin-pow']
LINK = ['-undefined', 'dynamic_lookup'] if sys.platform == 'darwin' else []
ROOT = Path(__file__).resolve().parent


def main():
    compiler = os.environ.get('CC', 'cc')
    build = ROOT / 'build' / 'compiled_ei'
    build.mkdir(parents=True, exist_ok=True)
    binary = build / ('_compiled_ei_kernel' + sysconfig.get_config_var('EXT_SUFFIX'))
    source = ROOT / 'compiled_ei_kernel.c'
    command = [compiler, *FLAGS, *LINK, '-I' + sysconfig.get_path('include'), str(source), '-o', str(binary), '-lm']
    subprocess.run(command, check=True, timeout=60)
    manifest = dict(schema=1, binary=binary.name, soabi=sysconfig.get_config_var('SOABI'),
                    numpy=numpy.__version__, scipy=scipy.__version__, flags=FLAGS, platform_link_flags=LINK,
                    command=command, compiler=subprocess.check_output([compiler, '--version'], text=True).splitlines()[0],
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (source, ROOT / 'compiled_ei.py', binary)})
    path = build / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(path)


if __name__ == '__main__':
    main()
