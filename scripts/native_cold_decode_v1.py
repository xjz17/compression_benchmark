#!/usr/bin/env python3
"""Independent native cold decode with the same worker CPU/memory boundary."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import native_prepared_cohort_v1 as data


def decode(cli: Path, frame: bytes, *, cpu: int | None = None,
           allow_large_memory: bool = True, timeout: float = 57600,
           limits: dict[str, int] | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix='native-cold-decode-') as directory:
        request=Path(directory)/'request.json'
        payload={'schema':'native-graph-request-v1',
            'frame_hex':frame.hex(),'allow_large_memory':allow_large_memory}
        if limits is not None:
            payload['limits']=limits
        request.write_bytes(data.canonical(payload))
        command=[str(cli.resolve()),'decode',str(request)]
        if cpu is not None:
            data.require(hasattr(os,'sched_getaffinity') and cpu in os.sched_getaffinity(0),
                         'cold decoder CPU unavailable')
            command=['taskset','-c',str(cpu),*command]
        if sys.platform.startswith('linux'):
            command=['prlimit','--as='+str(64<<30),'--',*command]
        # subprocess.run kills and reaps this owned, childless native process
        # on timeout; correctness verification is outside measurement clocks.
        result=subprocess.run(command,capture_output=True,timeout=timeout,check=False)
    data.require(result.returncode==0,'independent cold decode failed: '+
                 result.stderr.decode(errors='replace')[:2000])
    value=json.loads(result.stdout)
    data.require(value.get('schema')=='native-graph-response-v1','wrong cold decoder response schema')
    return value
