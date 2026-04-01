"""pcap_to_flow.py — extract per-flow features from raw QUIC captures.

Streams packets one-by-one from PCAP/PCAPNG files using Scapy's PcapReader
(NOT rdpcap, which would choke on our 2 GB+ files) and aggregates them into
bidirectional flows.  Each flow becomes one row in the output parquet file
with 13 statistical features + a binary label.

NOTE: Only UDP/IP packets are considered because QUIC sits on top of UDP.
      Non-UDP traffic (TCP handshakes, ICMP, ARP, ...) is silently skipped.
"""

from __future__ import annotations

import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scapy.all import PcapReader
from scapy.layers.inet import IP, UDP

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

FLOW_TIMEOUT_SEC  = 60.0       # close a flow after 60s of silence
FLUSH_EVERY       = 10_000     # check for timed-out flows every N packets
PROGRESS_EVERY    = 100_000    # log a status line every N packets
MAX_LIST_LEN      = 500        # cap IAT / pkt-size lists to keep RAM sane
PARQUET_BATCH_SIZE = 5_000     # write to disk once we have this many rows

# NOTE: might need to bump FLOW_TIMEOUT_SEC if we ever deal with
# slowloris-style QUIC traffic where flows are kept alive with tiny
# keepalive packets at very long intervals.

# ---------------------------------------------------------------------------
# Types & schema
# ---------------------------------------------------------------------------

FlowKey   = Tuple[str, str, int, int, int]   # (ip_a, ip_b, port_a, port_b, proto)
FlowState = Dict[str, object]

OUTPUT_COLS = [
    "flow_duration",
    "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
    "label",
]

PARQUET_SCHEMA = pa.schema([
    pa.field("flow_duration", pa.float64()),
    pa.field("fwd_pkts",      pa.int64()),
    pa.field("bwd_pkts",      pa.int64()),
    pa.field("fwd_bytes",     pa.int64()),
    pa.field("bwd_bytes",     pa.int64()),
    pa.field("iat_mean",      pa.float64()),
    pa.field("iat_std",       pa.float64()),
    pa.field("iat_max",       pa.float64()),
    pa.field("iat_min",       pa.float64()),
    pa.field("pkt_len_mean",  pa.float64()),
    pa.field("pkt_len_std",   pa.float64()),
    pa.field("pkt_len_max",   pa.float64()),
    pa.field("pkt_len_min",   pa.float64()),
    pa.field("label",         pa.int64()),
])

INT_COLS = {"fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "label"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_flow_key(src_ip: str, dst_ip: str,
                  src_port: int, dst_port: int, proto: int) -> FlowKey:
    """Canonical bidirectional key — sort so A->B and B->A hash the same."""
    fwd = (src_ip, dst_ip, src_port, dst_port)
    bwd = (dst_ip, src_ip, dst_port, src_port)
    c = min(fwd, bwd)
    return (c[0], c[1], c[2], c[3], proto)


def safe_stats(arr: List[float]) -> Tuple[float, float, float, float]:
    """(mean, std, max, min) — returns zeros for empty lists (single-pkt flows)."""
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def finalize_flow(flow: FlowState, label: int) -> Dict[str, object]:
    """Turn raw per-packet accumulators into the 13 statistical features."""
    iat_mean, iat_std, iat_max, iat_min = safe_stats(flow["iats"])
    pkt_mean, pkt_std, pkt_max, pkt_min = safe_stats(flow["packet_sizes"])
    duration_ms = (flow["last_ts"] - flow["first_ts"]) * 1000.0

    return {
        "flow_duration": duration_ms,
        "fwd_pkts":      flow["fwd_pkts"],
        "bwd_pkts":      flow["bwd_pkts"],
        "fwd_bytes":     flow["fwd_bytes"],
        "bwd_bytes":     flow["bwd_bytes"],
        "iat_mean": iat_mean, "iat_std": iat_std,
        "iat_max":  iat_max,  "iat_min": iat_min,
        "pkt_len_mean": pkt_mean, "pkt_len_std": pkt_std,
        "pkt_len_max":  pkt_max,  "pkt_len_min": pkt_min,
        "label": label,
    }


def flush_timed_out(flows: Dict[FlowKey, FlowState], current_ts: float,
                    buffer: List[dict], label: int) -> int:
    """Close flows idle for > FLOW_TIMEOUT_SEC."""
    expired = [k for k, v in flows.items()
               if (current_ts - v["last_ts"]) >= FLOW_TIMEOUT_SEC]
    for k in expired:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(expired)


def flush_all(flows: Dict[FlowKey, FlowState],
              buffer: List[dict], label: int) -> int:
    """Close every remaining flow — called at end of each file."""
    keys = list(flows.keys())
    for k in keys:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(keys)


def write_buffer(buffer: List[dict], writer: pq.ParquetWriter) -> None:
    """Dump buffered rows to parquet, enforcing strict dtypes."""
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=OUTPUT_COLS)
    for col in INT_COLS:
        df[col] = df[col].astype("int64")
    for col in set(OUTPUT_COLS) - INT_COLS:
        df[col] = df[col].astype("float64")
    writer.write_table(
        pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    )
    buffer.clear()


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def process_pcap(pcap_path: Path, label: int,
                 flows: Dict[FlowKey, FlowState],
                 buffer: List[dict],
                 writer: pq.ParquetWriter,
                 pkt_count: int) -> int:
    """Stream one pcap into the shared flow table.

    Uses PcapReader for constant-memory streaming — important because some
    of our benign captures are 2 GB each and rdpcap would OOM.
    """
    file_pkts = 0

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            file_pkts += 1
            pkt_count += 1

            if pkt_count % PROGRESS_EVERY == 0:
                log.info(
                    "  %s pkts | %s active flows | %s buffered",
                    f"{pkt_count:>12,}", f"{len(flows):>7,}", f"{len(buffer):>6,}",
                )

            # QUIC = UDP only; skip everything else
            if not (pkt.haslayer(IP) and pkt.haslayer(UDP)):
                continue

            ts       = float(pkt.time)
            src_ip   = pkt[IP].src
            dst_ip   = pkt[IP].dst
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
            proto    = int(pkt[IP].proto)
            pkt_size = float(pkt[IP].len)         # IP datagram, no L2 framing
            payload  = int(len(pkt[UDP].payload))  # QUIC payload only

            key = make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)

            if key not in flows:
                flows[key] = {
                    "first_ts": ts, "last_ts": ts,
                    "fwd_src_ip": src_ip, "fwd_src_port": src_port,
                    "fwd_pkts": 0, "bwd_pkts": 0,
                    "fwd_bytes": 0, "bwd_bytes": 0,
                    "packet_sizes": [], "iats": [],
                }

            flow = flows[key]

            # "forward" = whoever sent the first packet in this flow
            is_fwd = (src_ip == flow["fwd_src_ip"]
                      and src_port == flow["fwd_src_port"])

            if is_fwd:
                flow["fwd_pkts"]  += 1
                flow["fwd_bytes"] += payload
            else:
                flow["bwd_pkts"]  += 1
                flow["bwd_bytes"] += payload

            # IAT needs >=2 packets; capped to keep memory bounded
            if flow["packet_sizes"]:
                iat_ms = (ts - flow["last_ts"]) * 1000.0
                if len(flow["iats"]) < MAX_LIST_LEN:
                    flow["iats"].append(iat_ms)

            if len(flow["packet_sizes"]) < MAX_LIST_LEN:
                flow["packet_sizes"].append(pkt_size)

            flow["last_ts"] = ts

            if file_pkts % FLUSH_EVERY == 0:
                flush_timed_out(flows, ts, buffer, label)
                if len(buffer) >= PARQUET_BATCH_SIZE:
                    write_buffer(buffer, writer)

    return pkt_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="pcap -> parquet flow extractor")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent,
                        help="project root (default: script directory)")
    args = parser.parse_args()

    base       = args.base_dir.resolve()
    benign_dir = base / "data" / "raw_pcap" / "benign"
    ddos_dir   = base / "data" / "raw_pcap" / "ddos"
    out_dir    = base / "data" / "processed"
    out_path   = out_dir / "quic_dataset.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    benign_files = sorted(benign_dir.glob("*.pcap")) + sorted(benign_dir.glob("*.pcapng"))
    ddos_files   = sorted(ddos_dir.glob("*.pcap"))   + sorted(ddos_dir.glob("*.pcapng"))

    if not benign_files and not ddos_files:
        log.error("No pcap files found — check %s and %s", benign_dir, ddos_dir)
        sys.exit(1)

    log.info("Found %d benign + %d DDoS pcap(s)  ->  %s",
             len(benign_files), len(ddos_files), out_path)

    flows: Dict[FlowKey, FlowState] = {}
    buffer: List[dict] = []
    total_pkts = 0

    writer = pq.ParquetWriter(str(out_path), PARQUET_SCHEMA, compression="snappy")

    try:
        for pf in benign_files:
            log.info("[BENIGN] %s", pf.name)
            total_pkts = process_pcap(pf, label=0, flows=flows, buffer=buffer,
                                      writer=writer, pkt_count=total_pkts)
            n = flush_all(flows, buffer, label=0)
            log.info("  done — %s flows | buffer %s | %s pkts total",
                     f"{n:,}", f"{len(buffer):,}", f"{total_pkts:,}")
            if len(buffer) >= PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        for pf in ddos_files:
            log.info("[DDOS]   %s", pf.name)
            total_pkts = process_pcap(pf, label=1, flows=flows, buffer=buffer,
                                      writer=writer, pkt_count=total_pkts)
            n = flush_all(flows, buffer, label=1)
            log.info("  done — %s flows | buffer %s | %s pkts total",
                     f"{n:,}", f"{len(buffer):,}", f"{total_pkts:,}")
            if len(buffer) >= PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        write_buffer(buffer, writer)
    finally:
        writer.close()

    # quick sanity check on what we wrote
    result = pd.read_parquet(str(out_path), engine="pyarrow")
    log.info("=" * 56)
    log.info("  Packets processed : %12s", f"{total_pkts:,}")
    log.info("  Flows extracted   : %12s", f"{len(result):,}")
    for lbl, cnt in result["label"].value_counts().sort_index().items():
        tag = "benign" if lbl == 0 else "ddos  "
        log.info("    %s (%d) : %s", tag, lbl, f"{cnt:,}")
    log.info("  Saved to: %s", out_path)
    log.info("=" * 56)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
