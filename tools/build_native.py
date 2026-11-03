"""Build the optional local PyO3 extension, without changing system environments."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--toolchain', type=Path)
    p.add_argument('--vcvars', type=Path)
    p.add_argument('--test', action='store_true')
    p.add_argument('--planner-only', action='store_true', help='Install isolated planner module; leave the loaded input extension intact')
    args=p.parse_args()
    env=os.environ.copy()
    if args.vcvars:
        command=f'"{args.vcvars.resolve()}" >nul && set'
        output=subprocess.check_output('cmd.exe /d /s /c "'+command+'"',text=True,errors='replace')
        env.update(line.split('=',1) for line in output.splitlines() if '=' in line and not line.startswith('='))
    env['CARGO_HOME']=str(ROOT/'.cache/cargo')
    env['CARGO_TARGET_DIR']=str(ROOT/'.cache/rust-target')
    env['PYO3_PYTHON']=sys.executable
    if args.toolchain:
        env['PATH']=str(args.toolchain.resolve())+os.pathsep+env.get('PATH','')
        env['RUSTC']=str(args.toolchain.resolve()/'rustc.exe')
    cargo=str(args.toolchain/'cargo.exe') if args.toolchain else 'cargo'
    subprocess.run([cargo,'test' if args.test else 'build','--release','--manifest-path',str(ROOT/'native/Cargo.toml')],env=env,check=True)
    if not args.test:
        src=Path(env['CARGO_TARGET_DIR'])/'release/_native_backend.dll'
        if args.planner_only:
            planner_dir=ROOT/'th06ai/native_planner'
            planner_dir.mkdir(exist_ok=True)
            planner_dest=planner_dir/('_native_backend'+sysconfig.get_config_var('EXT_SUFFIX'))
            shutil.copy2(src,planner_dest)
            print(json.dumps({'planner_extension':str(planner_dest),'python':sys.executable}))
            return
        # Distinct ABI location leaves the DLL mapped by old workers intact.
        runtime_dir=ROOT/'th06ai/native_runtime'
        runtime_dir.mkdir(exist_ok=True)
        dest=runtime_dir/('_native_backend'+sysconfig.get_config_var('EXT_SUFFIX'))
        shutil.copy2(src,dest)
        print(json.dumps({'extension':str(dest),'python':sys.executable}))

if __name__=='__main__':main()
