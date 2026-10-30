#!/usr/bin/env python3
"""Stream LCBench run objects into small read-only-mappable replay arrays.

Each run is decoded separately. This reproduces the existing loader's float32
conversion, slice and division; it performs no surrogate fitting. PD1/ODBench
are decompressed one task at a time. Source and resulting array bytes are bound.
"""
import json
import mmap
from pathlib import Path

import numpy as np

from shared_temporal_design import ROOT,STAGING,TASKS,file_sha,canonical
from shared_temporal_resources import telemetry

DIRECTORY=STAGING/"replay_arrays"


def main():
    manifest_path=DIRECTORY/"manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_bytes())
        for name,pin in manifest["arrays"].items():
            if file_sha(DIRECTORY/name)!=pin:
                raise ValueError("replay array changed: "+name)
        for name,pin in manifest["sources"].items():
            if file_sha(ROOT/name)!=pin:
                raise ValueError("replay array source changed: "+name)
        print(json.dumps(dict(arrays="verified",**telemetry())),flush=True)
        return
    DIRECTORY.mkdir(exist_ok=True)
    arrays={}
    def save(benchmark,task,curves,hps):
        for kind,value in (("curves",curves),("hps",hps)):
            name=f"{benchmark}__{task}__{kind}.npy"
            path=DIRECTORY/name
            value=np.ascontiguousarray(value,dtype=np.float32)
            if path.exists():
                old=np.load(path,mmap_mode="r",allow_pickle=False)
                if old.dtype!=value.dtype or not np.array_equal(old,value):
                    raise ValueError("partial array differs")
            else:
                with path.open("xb") as stream:
                    np.save(stream,value,allow_pickle=False)
            arrays[name]=file_sha(path)
        print(json.dumps(dict(array_task=f"{benchmark}/{task}",**telemetry())),flush=True)
    from hp_geometry import _parse_json_object,_skip_json_value
    from hp_encoders import encode_lcbench
    source=ROOT/"datasets/lcbench/data_2k_lw.json"
    geometry=json.loads((ROOT/"staging/hp_geometry/v1/hp_geometry_v1.json").read_bytes())
    if file_sha(source)!=geometry["benchmarks"]["lcbench"]["source_file_sha256"]:
        raise ValueError("LCBench source hash differs")
    seen=[]
    with source.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        def task_member(task,start):
            if task not in TASKS["lcbench"]:
                raise ValueError("unexpected source task")
            curves,configs=[],[]
            def run_member(run,start):
                end=_skip_json_value(mapped,start)
                data=json.loads(mapped[start:end])
                values=data.get("log",{}).get("Train/val_accuracy",[])
                if len(values)>5:
                    curves.append(np.array(values[1:-1],dtype=np.float32)/100.)
                    configs.append(data.get("config",{}))
                return end
            end=_parse_json_object(mapped,start,run_member)
            if len(curves)!=2000 or any(len(c)!=50 for c in curves):
                raise ValueError("LCBench shape differs from fixed replay")
            save("lcbench",task,np.stack(curves),encode_lcbench(configs).astype(np.float32))
            seen.append(task)
            return end
        _parse_json_object(mapped,0,task_member)
    if set(seen)!=set(TASKS["lcbench"]):
        raise ValueError("incomplete LCBench task roster")
    from benchmark_design import PD1_PATH,PD1_CAPS,PD1_SHA256
    if file_sha(PD1_PATH)!=PD1_SHA256:
        raise ValueError("PD1 source hash differs")
    with np.load(PD1_PATH,allow_pickle=False) as archive:
        for task in TASKS["pd1"]:
            curves=archive[task].astype(np.float32)[:,:-1]
            if curves.shape!=(242,PD1_CAPS[task]):
                raise ValueError("PD1 cap differs")
            save("pd1",task,curves,archive[task+"__hps"].astype(np.float32))
    from cfbo_benchmark_io import CACHE_PATHS,TASKS as CF_TASKS
    odb= CACHE_PATHS["odbench"]
    geometry=json.loads((STAGING/"odbench_geometry.json").read_bytes())
    if file_sha(odb)!=geometry["cache_sha256"]:
        raise ValueError("ODBench cache differs")
    with np.load(odb,allow_pickle=False) as archive:
        for task in CF_TASKS["odbench"]:
            save("odbench",task,archive[task],archive[task+"__hps"][:,geometry["geometry"]["active_dimension_indices"]])
    manifest=dict(schema="shared-temporal-replay-arrays/v1",arrays=arrays,
        sources={str(p.relative_to(ROOT)):file_sha(p) for p in (source,PD1_PATH,odb)},
        builder_sha256=file_sha(Path(__file__)),transforms="exact existing float32 loader operations; no fitting",
        memory=telemetry())
    with manifest_path.open("xb") as stream:
        stream.write(canonical(manifest)+b"\n")


if __name__=="__main__":
    main()
