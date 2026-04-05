#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Any

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_NAMES = {
    0: "f32",
    1: "f16",
    8: "q8_0",
    16: "i8",
}

KV_TYPE_NAMES = {
    GGUF_TYPE_UINT8: "u8",
    GGUF_TYPE_INT8: "i8",
    GGUF_TYPE_UINT16: "u16",
    GGUF_TYPE_INT16: "i16",
    GGUF_TYPE_UINT32: "u32",
    GGUF_TYPE_INT32: "i32",
    GGUF_TYPE_FLOAT32: "f32",
    GGUF_TYPE_BOOL: "bool",
    GGUF_TYPE_STRING: "string",
    GGUF_TYPE_ARRAY: "array",
    GGUF_TYPE_UINT64: "u64",
    GGUF_TYPE_INT64: "i64",
    GGUF_TYPE_FLOAT64: "f64",
}


@dataclass
class TensorInfo:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int


@dataclass
class GGUFMeta:
    path: Path
    version: int
    tensor_count: int
    kv_count: int
    alignment: int
    kvs: dict[str, Any]
    tensors: list[TensorInfo]


class Reader:
    def __init__(self, fp: BinaryIO):
        self.fp = fp

    def read_exact(self, n: int) -> bytes:
        data = self.fp.read(n)
        if len(data) != n:
            raise EOFError(f"unexpected EOF while reading {n} bytes")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read_exact(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read_exact(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read_exact(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.read_exact(8))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.read_exact(4))[0]

    def f64(self) -> float:
        return struct.unpack("<d", self.read_exact(8))[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.read_exact(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.read_exact(2))[0]

    def u8(self) -> int:
        return struct.unpack("<B", self.read_exact(1))[0]

    def i8(self) -> int:
        return struct.unpack("<b", self.read_exact(1))[0]

    def bool(self) -> bool:
        return bool(struct.unpack("<b", self.read_exact(1))[0])

    def string(self) -> str:
        n = self.u64()
        return self.read_exact(n).decode("utf-8")

    def value(self, typ: int) -> Any:
        if typ == GGUF_TYPE_UINT8:
            return self.u8()
        if typ == GGUF_TYPE_INT8:
            return self.i8()
        if typ == GGUF_TYPE_UINT16:
            return self.u16()
        if typ == GGUF_TYPE_INT16:
            return self.i16()
        if typ == GGUF_TYPE_UINT32:
            return self.u32()
        if typ == GGUF_TYPE_INT32:
            return self.i32()
        if typ == GGUF_TYPE_FLOAT32:
            return self.f32()
        if typ == GGUF_TYPE_BOOL:
            return self.bool()
        if typ == GGUF_TYPE_STRING:
            return self.string()
        if typ == GGUF_TYPE_UINT64:
            return self.u64()
        if typ == GGUF_TYPE_INT64:
            return self.i64()
        if typ == GGUF_TYPE_FLOAT64:
            return self.f64()
        if typ == GGUF_TYPE_ARRAY:
            elem_type = self.u32()
            n = self.u64()
            return {
                "type": KV_TYPE_NAMES.get(elem_type, str(elem_type)),
                "items": [self.value(elem_type) for _ in range(n)],
            }
        raise ValueError(f"unsupported GGUF kv type: {typ}")


def parse_gguf(path: Path) -> GGUFMeta:
    with path.open("rb") as fp:
        r = Reader(fp)
        magic = r.read_exact(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file")

        version = r.u32()
        tensor_count = r.u64()
        kv_count = r.u64()

        kvs: dict[str, Any] = {}
        for _ in range(kv_count):
            key = r.string()
            typ = r.u32()
            kvs[key] = r.value(typ)

        tensors: list[TensorInfo] = []
        for _ in range(tensor_count):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ggml_type = r.u32()
            offset = r.u64()
            tensors.append(TensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset))

        alignment = int(kvs.get("general.alignment", 32))
        return GGUFMeta(
            path=path,
            version=version,
            tensor_count=tensor_count,
            kv_count=kv_count,
            alignment=alignment,
            kvs=kvs,
            tensors=tensors,
        )


def format_shape(dims: list[int]) -> str:
    return "x".join(str(x) for x in dims)


def format_tensor_line(t: TensorInfo, *, mark_rknpu: bool) -> str:
    prefix = "[RKNPU]" if mark_rknpu else "       "
    ggml_type = GGML_TYPE_NAMES.get(t.ggml_type, str(t.ggml_type))
    return f"{prefix} {t.name:<48} type={ggml_type:<6} shape={format_shape(t.dims):<20} offset={t.offset}"


def summarize(meta: GGUFMeta) -> None:
    print(f"file: {meta.path}")
    print(f"version: {meta.version}")
    print(f"tensors: {meta.tensor_count}")
    print(f"kv pairs: {meta.kv_count}")
    print(f"alignment: {meta.alignment}")
    print()

    rknpu_tensors = [t for t in meta.tensors if t.name.endswith('.__rknpu_blob')]
    regular_weights = [t for t in meta.tensors if t.name.endswith('.weight') and 'norm' not in t.name]
    rknpu_kvs = sorted(k for k in meta.kvs if k.startswith('rknpu.'))

    print(f"regular non-norm weights: {len(regular_weights)}")
    print(f"RKNPU blob tensors: {len(rknpu_tensors)}")
    print(f"RKNPU metadata keys: {len(rknpu_kvs)}")
    print()

    print("all tensor entries:")
    for tensor in meta.tensors:
        print(format_tensor_line(tensor, mark_rknpu=tensor.name.endswith('.__rknpu_blob')))

    if rknpu_kvs:
        print()
        print("RKNPU metadata:")
        for key in rknpu_kvs:
            print(f"[RKNPU] {key} = {meta.kvs[key]}")


def diff(before: GGUFMeta, after: GGUFMeta) -> None:
    before_tensors = {t.name: t for t in before.tensors}
    after_tensors = {t.name: t for t in after.tensors}

    added_names = sorted(set(after_tensors) - set(before_tensors))
    removed_names = sorted(set(before_tensors) - set(after_tensors))
    common_names = sorted(set(before_tensors) & set(after_tensors))

    before_kvs = set(before.kvs)
    after_kvs = set(after.kvs)
    added_kvs = sorted(after_kvs - before_kvs)
    removed_kvs = sorted(before_kvs - after_kvs)
    changed_kvs = sorted(k for k in before_kvs & after_kvs if before.kvs[k] != after.kvs[k])

    print(f"before: {before.path}")
    print(f"after : {after.path}")
    print()
    print(f"tensor count: {before.tensor_count} -> {after.tensor_count}  (delta {after.tensor_count - before.tensor_count:+d})")
    print(f"kv count    : {before.kv_count} -> {after.kv_count}  (delta {after.kv_count - before.kv_count:+d})")
    print()

    print(f"added tensors   : {len(added_names)}")
    print(f"removed tensors : {len(removed_names)}")
    print(f"added kvs       : {len(added_kvs)}")
    print(f"changed kvs     : {len(changed_kvs)}")
    print()

    if added_names:
        print("added tensors:")
        for name in added_names:
            tensor = after_tensors[name]
            is_rknpu = name.endswith('.__rknpu_blob')
            prefix = "[ADDED][RKNPU]" if is_rknpu else "[ADDED]       "
            ggml_type = GGML_TYPE_NAMES.get(tensor.ggml_type, str(tensor.ggml_type))
            print(f"{prefix} {name:<48} type={ggml_type:<6} shape={format_shape(tensor.dims):<20} offset={tensor.offset}")
        print()

    if added_kvs:
        print("added kvs:")
        for key in added_kvs:
            prefix = "[ADDED][RKNPU]" if key.startswith('rknpu.') else "[ADDED]       "
            print(f"{prefix} {key} = {after.kvs[key]}")
        print()

    if changed_kvs:
        print("changed kvs:")
        for key in changed_kvs:
            prefix = "[CHANGED][RKNPU]" if key.startswith('rknpu.') else "[CHANGED]       "
            print(f"{prefix} {key}")
            print(f"    before: {before.kvs[key]}")
            print(f"    after : {after.kvs[key]}")
        print()

    if removed_names:
        print("removed tensors:")
        for name in removed_names:
            print(f"[REMOVED] {name}")
        print()

    if removed_kvs:
        print("removed kvs:")
        for key in removed_kvs:
            print(f"[REMOVED] {key}")
        print()

    if not added_names and not removed_names and not added_kvs and not changed_kvs and not removed_kvs:
        print("no metadata-level changes detected")

    added_rknpu = [name for name in added_names if name.endswith('.__rknpu_blob')]
    if added_rknpu:
        print(f"summary: detected {len(added_rknpu)} newly added RKNPU blob tensors")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect GGUF tensor metadata and highlight RKNPU-added content")
    parser.add_argument("gguf", nargs="+", help="one GGUF file to inspect, or two GGUF files to diff")
    args = parser.parse_args()

    if len(args.gguf) not in (1, 2):
        parser.error("provide one GGUF file to inspect or two GGUF files to diff")

    paths = [Path(p).expanduser() for p in args.gguf]
    metas = [parse_gguf(path) for path in paths]

    if len(metas) == 1:
        summarize(metas[0])
    else:
        diff(metas[0], metas[1])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        pass
