"""A111 native process bridge and typed-port framing, with no Python codec kernel.

The supplied b controls the low bits; it is not a FastPFor width selector.
EHP1 composes A111 with A020 and is not an upstream FastPFor packet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import subprocess

ALGORITHM_NAME = "op.outlier_separation.high_bits.uint32.cpp_v1"
SOURCE = "cpp/src/exception_separation.hpp"
BINARY = Path(__file__).resolve().parents[1] / "cpp/build/exception_separation_cli"
HEADER = struct.Struct("<4sBII")


def run(command: str, data: bytes, width: int | None = None) -> bytes:
    if not BINARY.is_file():
        raise RuntimeError(f"missing C++ action binary {BINARY}; build exception_separation_cli")
    args = [str(BINARY), command]
    if width is not None:
        if type(width) is not int or not 0 <= width <= 32:
            raise ValueError("exception separation width must be an integer in [0,32]")
        args.append(str(width))
    result = subprocess.run(args, input=bytes(data), capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace").strip())
    return result.stdout


def forward(raw_uint32_le: bytes, width: int) -> bytes:
    return run("forward", raw_uint32_le, width)


def inverse(packet: bytes) -> bytes:
    return run("inverse", packet)


@dataclass(frozen=True)
class Ports:
    """All three payloads are independently addressable uint32 LE streams."""

    width: int
    count: int
    low: bytes
    positions: bytes
    high: bytes


def to_ports(packet: bytes) -> Ports:
    packet = run("validate", packet)
    _, width, count, exceptions = HEADER.unpack_from(packet)
    low_end = HEADER.size + count * 4
    positions_end = low_end + exceptions * 4
    return Ports(width, count, packet[HEADER.size:low_end],
                 packet[low_end:positions_end], packet[positions_end:])


def from_ports(ports: Ports) -> bytes:
    if type(ports.width) is not int or not 0 <= ports.width <= 32:
        raise ValueError("exception separation width must be an integer in [0,32]")
    if type(ports.count) is not int or not 0 <= ports.count <= 0xffffffff:
        raise ValueError("exception separation count must fit uint32")
    if (len(ports.low) != ports.count * 4 or len(ports.positions) % 4
            or len(ports.high) != len(ports.positions)):
        raise ValueError("exception port lengths do not match metadata")
    exceptions = len(ports.positions) // 4
    if exceptions > ports.count:
        raise ValueError("too many exception positions")
    packet = HEADER.pack(b"EHS1", ports.width, ports.count, exceptions)
    packet += bytes(ports.low) + bytes(ports.positions) + bytes(ports.high)
    return run("validate", packet)


def pack(packet: bytes) -> bytes:
    """A020 on low(b), positions(max width), and high(max width)."""
    return run("pack", packet)


def unpack(packet: bytes) -> bytes:
    return run("unpack", packet)
