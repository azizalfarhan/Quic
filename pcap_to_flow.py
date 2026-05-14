"""Extract per-flow features from raw QUIC captures.

Streams packets one-by-one with Scapy's PcapReader (NOT rdpcap — the
benign captures hit 2+ GB and rdpcap loads everything into RAM at once)
and aggregates them into bidirectional flows. Each flow becomes one row
in the output parquet with the 13 statistical features in
config.FEATURE_COLS, a binary label, and a `source` tag.

QUIC = UDP, so any non-UDP traffic (TCP handshakes, ICMP, ARP, ...) is
silently dropped.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scapy.all import PcapReader
from scapy.layers.inet import IP, UDP

import config

config.configure_logging()
log = logging.getLogger(__name__)


FlowKey   = tuple[str, str, int, int, int]
FlowState = dict[str, object]

# SCIENTIFIC NOTE: IP addresses, MACs, and Ports are strictly excluded from the final features to prevent Identifier-Leakage. This forces the model to learn the actual behavioral patterns (packet sizes, timing) of a DDoS attack rather than memorizing attacker IPs, ensuring real-world generalizability.
OUTPUT_COLS = [*config.FEATURE_COLS, config.LABEL_COL, config.SOURCE_COL]

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
    pa.field(config.LABEL_COL,  pa.int64()),
    pa.field(config.SOURCE_COL, pa.string()),
])

INT_COLS = {"fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", config.LABEL_COL}
STR_COLS = {config.SOURCE_COL}


def make_flow_key(src_ip: str, dst_ip: str,
                  src_port: int, dst_port: int, proto: int) -> FlowKey:
    # Canonical key so A->B and B->A hash to the same flow.
    fwd = (src_ip, dst_ip, src_port, dst_port)
    bwd = (dst_ip, src_ip, dst_port, src_port)
    c = min(fwd, bwd)
    return (c[0], c[1], c[2], c[3], proto)


def safe_stats(arr: list[float]) -> tuple[float, float, float, float]:
    # Empty list = single-packet flow; treat as zeros instead of NaN so
    # downstream models don't need to special-case it.
    if not arr:
        return 0.0, 0.0, 0.0, 0.0
    a = np.asarray(arr, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.max()), float(a.min())


def finalize_flow(flow: FlowState, label: int) -> dict[str, object]:
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
        config.LABEL_COL:  label,
        config.SOURCE_COL: flow["source"],
    }


def flush_timed_out(flows: dict[FlowKey, FlowState], current_ts: float,
                    buffer: list[dict], label: int) -> int:
    expired = [k for k, v in flows.items()
               if (current_ts - v["last_ts"]) >= config.FLOW_TIMEOUT_SEC]
    for k in expired:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(expired)


def flush_all(flows: dict[FlowKey, FlowState],
              buffer: list[dict], label: int) -> int:
    keys = list(flows.keys())
    for k in keys:
        buffer.append(finalize_flow(flows.pop(k), label))
    return len(keys)


def detect_source(pcap_path: Path) -> str:
    # Maps the parent folder name to a source tag (kaggle / client / original).
    # Anything outside the known set is "original" so old Phase 2 captures
    # sitting directly under benign/ or ddos/ keep working unchanged.
    for part in pcap_path.parts:
        if part in config.KNOWN_SOURCES:
            return part
    return "original"


def write_buffer(buffer: list[dict], writer: pq.ParquetWriter) -> None:
    if not buffer:
        return
    df = pd.DataFrame(buffer, columns=OUTPUT_COLS)
    for col in INT_COLS:
        df[col] = df[col].astype("int64")
    for col in STR_COLS:
        df[col] = df[col].astype("string")
    for col in set(OUTPUT_COLS) - INT_COLS - STR_COLS:
        df[col] = df[col].astype("float64")
    writer.write_table(
        pa.Table.from_pandas(df, schema=PARQUET_SCHEMA, preserve_index=False)
    )
    buffer.clear()


def process_pcap(pcap_path: Path, label: int,
                 flows: dict[FlowKey, FlowState],
                 buffer: list[dict],
                 writer: pq.ParquetWriter,
                 pkt_count: int) -> int:
    file_pkts = 0
    source = detect_source(pcap_path)

    with PcapReader(str(pcap_path)) as reader:
        for pkt in reader:
            file_pkts += 1
            pkt_count += 1

            if pkt_count % config.PROGRESS_EVERY == 0:
                log.info(
                    "  %s pkts | %s active flows | %s buffered",
                    f"{pkt_count:>12,}", f"{len(flows):>7,}", f"{len(buffer):>6,}",
                )

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
                    "source": source,
                }

            flow = flows[key]
            is_fwd = (src_ip == flow["fwd_src_ip"]
                      and src_port == flow["fwd_src_port"])

            if is_fwd:
                flow["fwd_pkts"]  += 1
                flow["fwd_bytes"] += payload
            else:
                flow["bwd_pkts"]  += 1
                flow["bwd_bytes"] += payload

            # IAT needs >=2 packets. Caps keep RAM bounded on flooded flows
            # where the attacker fires millions of packets to the same key.
            if flow["packet_sizes"]:
                iat_ms = (ts - flow["last_ts"]) * 1000.0
                if len(flow["iats"]) < config.MAX_LIST_LEN:
                    flow["iats"].append(iat_ms)

            if len(flow["packet_sizes"]) < config.MAX_LIST_LEN:
                flow["packet_sizes"].append(pkt_size)

            flow["last_ts"] = ts

            if file_pkts % config.FLUSH_EVERY == 0:
                flush_timed_out(flows, ts, buffer, label)
                if len(buffer) >= config.PARQUET_BATCH_SIZE:
                    write_buffer(buffer, writer)

    return pkt_count


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pcap -> parquet flow extractor")
    p.add_argument("--benign-dir", type=Path, default=config.BENIGN_DIR)
    p.add_argument("--ddos-dir",   type=Path, default=config.DDOS_DIR)
    p.add_argument("--out-path",   type=Path, default=config.PARQUET_PATH)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # rglob picks up the per-source subfolders (kaggle/, client/, original/)
    # introduced in Phase 3.
    def _collect(d: Path) -> list[Path]:
        return sorted(d.rglob("*.pcap")) + sorted(d.rglob("*.pcapng"))

    benign_files = _collect(args.benign_dir)
    ddos_files   = _collect(args.ddos_dir)

    if not benign_files and not ddos_files:
        log.error(
            "No pcap files found — check %s and %s",
            args.benign_dir, args.ddos_dir,
        )
        sys.exit(1)

    log.info("Found %d benign + %d DDoS pcap(s)  ->  %s",
             len(benign_files), len(ddos_files), args.out_path)

    flows: dict[FlowKey, FlowState] = {}
    buffer: list[dict] = []
    total_pkts = 0

    writer = pq.ParquetWriter(str(args.out_path), PARQUET_SCHEMA, compression="snappy")

    try:
        for pf in benign_files:
            log.info("[BENIGN] %s", pf.name)
            total_pkts = process_pcap(pf, label=0, flows=flows, buffer=buffer,
                                      writer=writer, pkt_count=total_pkts)
            n = flush_all(flows, buffer, label=0)
            log.info("  done — %s flows | buffer %s | %s pkts total",
                     f"{n:,}", f"{len(buffer):,}", f"{total_pkts:,}")
            if len(buffer) >= config.PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        for pf in ddos_files:
            log.info("[DDOS]   %s", pf.name)
            total_pkts = process_pcap(pf, label=1, flows=flows, buffer=buffer,
                                      writer=writer, pkt_count=total_pkts)
            n = flush_all(flows, buffer, label=1)
            log.info("  done — %s flows | buffer %s | %s pkts total",
                     f"{n:,}", f"{len(buffer):,}", f"{total_pkts:,}")
            if len(buffer) >= config.PARQUET_BATCH_SIZE:
                write_buffer(buffer, writer)

        write_buffer(buffer, writer)
    finally:
        writer.close()

    result = pd.read_parquet(str(args.out_path), engine="pyarrow")
    log.info("=" * 56)
    log.info("  Packets processed : %12s", f"{total_pkts:,}")
    log.info("  Flows extracted   : %12s", f"{len(result):,}")
    for lbl, cnt in result[config.LABEL_COL].value_counts().sort_index().items():
        tag = "benign" if lbl == 0 else "ddos  "
        log.info("    %s (%d) : %s", tag, lbl, f"{cnt:,}")
    log.info("  By source x label:")
    breakdown = result.groupby([config.SOURCE_COL, config.LABEL_COL]).size()
    for (src, lbl), cnt in breakdown.items():
        tag = "benign" if lbl == 0 else "ddos"
        log.info("    %-10s %-6s : %s", src, tag, f"{cnt:,}")
    log.info("  Saved to: %s", args.out_path)
    log.info("=" * 56)


if __name__ == "__main__":
    main()
