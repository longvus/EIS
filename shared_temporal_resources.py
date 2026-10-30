"""Bounded scheduling and memory telemetry for episode runs.
"""

from pathlib import Path

import resource

def available_bytes():
    fields={line.split(":",1)[0]:line.split(":",1)[1].strip() for line in Path("/proc/meminfo").read_text().splitlines()}
    return int(fields["MemAvailable"].split()[0])*1024

def telemetry():
    """Memory telemetry for logs and manifests; Linux reads /proc, other platforms report the peak only."""
    try:
        status=Path("/proc/self/status").read_text().splitlines()
        rss=next(int(line.split()[1]) for line in status if line.startswith("VmRSS:"))
        available=available_bytes()//1024**2
    except OSError:
        rss,available=None,None
    return dict(rss_kib=rss,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,available_mib=available)
