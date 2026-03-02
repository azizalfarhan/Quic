# pcap_to_flow.py
# reads pcap files (benign + ddos) and extracts per-flow features into a parquet dataset
# uses PcapReader so it streams packets one by one — needed because files can be 2GB+

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scapy.all import PcapReader
from scapy.layers.inet import IP, UDP

# constants
FLOW_TIMEOUT_SEC = 60.0      # close a flow after 60s of inactivity
FLUSH_EVERY_N_PKTS = 10_000  # how often to check for timed-out flows
PROGRESS_EVERY_N_PKTS = 100_000
MAX_LIST_LEN = 500           # cap lists per flow to keep RAM under control
PARQUET_BATCH_SIZE = 5_000   # write to parquet in chunks, not all at once

# type alias for the bidirectional flow key
FlowKey = Tuple[str, str, int, int, int]  # (ip_a, ip_b, port_a, port_b, proto)

# all columns written to the parquet file (features + label)
# note: train_baseline.py uses its own FEATURE_COLS that excludes "label"
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


def make_flow_key(src_ip, dst_ip, src_port, dst_port, proto):
    # bidirectional key: sort the 5-tuple so A->B and B->A hash to the same entry
    fwd = (src_ip, dst_ip, src_port, dst_port)
    bwd = (dst_ip, src_ip, dst_port, src_port)
    c = min(fwd, bwd)
    return (c[0], c[1], c[2], c[3], proto)


def safe_stats(arr):
    # returns (mean, std, max, min) — zeros if the list is empty (e.g. single-packet flow)
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def finalize_flow(flow, label):
    # compute all features from the accumulated flow state
    iat_mean, iat_std, iat_max, iat_min = safe_stats(flow["iats"])
    pkt_mean, pkt_std, pkt_max, pkt_min = safe_stats(flow["packet_sizes"])
    duration_ms = (flow["last_ts"] - flow["first_ts"]) * 1000.0
    # TODO: maybe also add total_pkts = fwd_pkts + bwd_pkts as a feature later

    return {
        "flow_duration": duration_ms,
        "fwd_pkts":      flow["fwd_pkts"],
        "bwd_pkts":      flow["bwd_pkts"],
        "fwd_bytes":     flow["fwd_bytes"],
        "bwd_bytes":     flow["bwd_bytes"],
        "iat_mean":      iat_mean,
        "iat_std":       iat_std,
        "iat_max":       iat_max,
        "iat_min":       iat_min,
        "pkt_len_mean":  pkt_mean,
        "pkt_len_std":   pkt_std,
        "pkt_len_max":   pkt_max,
        "pkt_len_min":   pkt_min,
        "label":         label,
    }


def flush_timed_out(flows, current_ts, buffer, label):
    timed_out = [
        k for k, v in flows.items()
        if (current_ts - v["last_ts"]) >= FLOW_TIMEOUT_SEC
    ]
    for k in timed_out:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(timed_out)


def flush_all_remaining(flows, buffer, label):
    # called at end of each file to close all still-open flows
    keys = list(flows.keys())
    for k in keys:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(keys)


def write_buffer(buffer, writer):
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=OUTPUT_COLS)
    for col in ("fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "label"):
        df[col] = df[col].astype("int64")
    float_cols = [c for c in OUTPUT_COLS if c not in
                  ("fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "label")]
    for col in float_cols:
        df[col] = df[col].astype("float64")
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    writer.write_table(table)
    buffer.clear()


def process_pcap(pcap_path, label, flows, buffer, writer, global_pkt_count):
    # stream packets one at a time — do NOT use rdpcap() here, files are up to 2GB
    file_pkt_count = 0

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            file_pkt_count += 1
            global_pkt_count += 1

            if global_pkt_count % PROGRESS_EVERY_N_PKTS == 0:
                print(
                    f"  [PROGRESS] {global_pkt_count:>12,} packets total | "
                    f"{len(flows):>7,} active flows | "
                    f"{len(buffer):>6,} buffered records"
                )

            # only care about UDP/IP (QUIC runs over UDP)
            if not (pkt.haslayer(IP) and pkt.haslayer(UDP)):
                continue

            ts       = float(pkt.time)
            src_ip   = pkt[IP].src
            dst_ip   = pkt[IP].dst
            src_port = int(pkt[UDP].sport)
            dst_port = int(pkt[UDP].dport)
            # proto is always 17 here (UDP-only filter above), kept for standard 5-tuple
            proto    = int(pkt[IP].proto)
            # pkt_size: IP datagram length from IP header — OS/NIC independent, no FCS
            # NOTE: pkt_len_* features measure IP datagram size;
            #       fwd_bytes/bwd_bytes measure UDP payload only (different units, by design)
            pkt_size = float(pkt[IP].len)
            # payload = QUIC data only (UDP payload, excludes all headers)
            payload  = int(len(pkt[UDP].payload))

            key = make_flow_key(src_ip, dst_ip, src_port, dst_port, proto)

            if key not in flows:
                flows[key] = {
                    "first_ts":     ts,
                    "last_ts":      ts,  # single timestamp field — used for timeout, IAT, and duration
                    "fwd_src_ip":   src_ip,
                    "fwd_src_port": src_port,
                    "fwd_pkts":     0,
                    "bwd_pkts":     0,
                    "fwd_bytes":    0,
                    "bwd_bytes":    0,
                    "packet_sizes": [],
                    "iats":         [],
                }

            flow = flows[key]

            # "forward" = direction of the first observed packet (capture-order-relative)
            # fwd_pkts/fwd_bytes always correspond to the initiating sender
            is_fwd = (src_ip == flow["fwd_src_ip"] and src_port == flow["fwd_src_port"])

            if is_fwd:
                flow["fwd_pkts"]  += 1
                flow["fwd_bytes"] += payload
            else:
                flow["bwd_pkts"]  += 1
                flow["bwd_bytes"] += payload

            # IAT: skip first packet (no previous timestamp yet)
            # for flows with N packets this gives exactly N-1 IAT values
            # capping at MAX_LIST_LEN to keep memory under control —
            # for huge flows this means stats are only from the first ~500 packets
            if flow["packet_sizes"]:
                iat_ms = (ts - flow["last_ts"]) * 1000.0
                if len(flow["iats"]) < MAX_LIST_LEN:
                    flow["iats"].append(iat_ms)
            # print(f"flow {key}: iats={flow['iats'][-3:]}")  # debug

            if len(flow["packet_sizes"]) < MAX_LIST_LEN:
                flow["packet_sizes"].append(pkt_size)

            flow["last_ts"] = ts

            if file_pkt_count % FLUSH_EVERY_N_PKTS == 0:
                flush_timed_out(flows, ts, buffer, label)
                if len(buffer) >= PARQUET_BATCH_SIZE:
                    write_buffer(buffer, writer)

    return global_pkt_count


def main():
    parser = argparse.ArgumentParser(description="pcap -> parquet flow feature extractor")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="project root directory",
    )
    args = parser.parse_args()

    base       = args.base_dir.resolve()
    benign_dir = base / "data" / "raw_pcap" / "benign"
    ddos_dir   = base / "data" / "raw_pcap" / "ddos"
    out_dir    = base / "data" / "processed"
    out_path   = out_dir / "quic_dataset.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)

    benign_files = sorted(benign_dir.glob("*.pcap")) + sorted(benign_dir.glob("*.pcapng"))
    ddos_files   = sorted(ddos_dir.glob("*.pcap"))   + sorted(ddos_dir.glob("*.pcapng"))
    # print("benign files:", benign_files)
    # print("ddos files:", ddos_files)

    if not benign_files and not ddos_files:
        print("[ERROR] No pcap files found.")
        print(f"  Place benign .pcap files in : {benign_dir}")
        print(f"  Place DDoS  .pcap files in  : {ddos_dir}")
        sys.exit(1)

    print(f"Found {len(benign_files)} benign + {len(ddos_files)} DDoS pcap file(s)")
    print(f"Output: {out_path}\n")

    flows  = {}
    buffer = []
    global_pkt_count = 0

    writer = pq.ParquetWriter(str(out_path), PARQUET_SCHEMA, compression="snappy")

    try:
        for pcap_file in benign_files:
            print(f"[BENIGN] {pcap_file.name}")
            global_pkt_count = process_pcap(
                pcap_file, label=0,
                flows=flows, buffer=buffer,
                writer=writer,
                global_pkt_count=global_pkt_count,
            )
            flushed = flush_all_remaining(flows, buffer, label=0)
            print(
                f"  Done — {flushed:,} flows closed | "
                f"buffer: {len(buffer):,} | "
                f"total pkts: {global_pkt_count:,}"
            )
            if len(buffer) >= PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        for pcap_file in ddos_files:
            print(f"[DDOS]   {pcap_file.name}")
            global_pkt_count = process_pcap(
                pcap_file, label=1,
                flows=flows, buffer=buffer,
                writer=writer,
                global_pkt_count=global_pkt_count,
            )
            flushed = flush_all_remaining(flows, buffer, label=1)
            print(
                f"  Done — {flushed:,} flows closed | "
                f"buffer: {len(buffer):,} | "
                f"total pkts: {global_pkt_count:,}"
            )
            if len(buffer) >= PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        write_buffer(buffer, writer)

    finally:
        writer.close()

    # quick summary
    final_df = pd.read_parquet(str(out_path), engine="pyarrow")
    print()
    print("=" * 56)
    print(f"  Total packets processed : {global_pkt_count:>12,}")
    print(f"  Total flows extracted   : {len(final_df):>12,}")
    print(f"  Label distribution:")
    for lbl, cnt in final_df["label"].value_counts().sort_index().items():
        name = "benign (0)" if lbl == 0 else "ddos   (1)"
        print(f"    {name} : {cnt:,}")
    print(f"  Output saved to         : {out_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
