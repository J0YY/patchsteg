"""Compact UTF-8 payload framing for higher-capacity steganographic channels."""
from __future__ import annotations

from dataclasses import dataclass
import struct
import zlib


@dataclass
class PayloadPacket:
    bits: list[int]
    compressed: bool
    original_bytes: int
    stored_bytes: int
    crc32: int
    header_bits: int

    @property
    def total_bits(self) -> int:
        return len(self.bits)

    @property
    def body_bits(self) -> int:
        return len(self.bits) - self.header_bits


@dataclass
class DecodedPayload:
    success: bool
    complete: bool
    text: str | None
    error: str | None
    compressed: bool
    original_bytes: int
    stored_bytes: int
    crc32: int | None
    header_bits: int
    total_bits: int

    @property
    def body_bits(self) -> int:
        return max(0, self.total_bits - self.header_bits)


class PayloadCodec:
    """Frame, compress, and validate arbitrary UTF-8 text payloads."""

    MAGIC = b"PS"
    VERSION = 1
    FLAG_COMPRESSED = 1
    HEADER_FORMAT = ">2sBBIII"
    HEADER_BYTES = struct.calcsize(HEADER_FORMAT)

    @classmethod
    def header_bits(cls) -> int:
        return cls.HEADER_BYTES * 8

    @staticmethod
    def bytes_to_bits(data: bytes) -> list[int]:
        bits: list[int] = []
        for byte in data:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits

    @staticmethod
    def bits_to_bytes(bits: list[int]) -> bytes:
        if len(bits) % 8 != 0:
            raise ValueError("Bit stream length must be a multiple of 8")
        out = bytearray()
        for start in range(0, len(bits), 8):
            value = 0
            for bit in bits[start:start + 8]:
                value = (value << 1) | int(bit)
            out.append(value)
        return bytes(out)

    @classmethod
    def pack_text(
        cls,
        text: str,
        *,
        encoding: str = "utf-8",
        compression_level: int = 9,
        enable_compression: bool = True,
    ) -> PayloadPacket:
        raw = text.encode(encoding)
        compressed = zlib.compress(raw, level=compression_level) if enable_compression else raw
        use_compressed = enable_compression and len(compressed) < len(raw)
        body = compressed if use_compressed else raw
        flags = cls.FLAG_COMPRESSED if use_compressed else 0
        crc32 = zlib.crc32(body) & 0xFFFFFFFF
        header = struct.pack(
            cls.HEADER_FORMAT,
            cls.MAGIC,
            cls.VERSION,
            flags,
            len(raw),
            len(body),
            crc32,
        )
        bits = cls.bytes_to_bits(header + body)
        return PayloadPacket(
            bits=bits,
            compressed=use_compressed,
            original_bytes=len(raw),
            stored_bytes=len(body),
            crc32=crc32,
            header_bits=cls.header_bits(),
        )

    @classmethod
    def unpack_bits(
        cls,
        bits: list[int],
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> DecodedPayload:
        if len(bits) < cls.header_bits():
            return DecodedPayload(
                success=False,
                complete=False,
                text=None,
                error="Incomplete header",
                compressed=False,
                original_bytes=0,
                stored_bytes=0,
                crc32=None,
                header_bits=cls.header_bits(),
                total_bits=cls.header_bits(),
            )

        header_bytes = cls.bits_to_bytes(bits[:cls.header_bits()])
        try:
            magic, version, flags, original_bytes, stored_bytes, crc32 = struct.unpack(
                cls.HEADER_FORMAT, header_bytes
            )
        except struct.error as exc:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error=f"Header parse failed: {exc}",
                compressed=False,
                original_bytes=0,
                stored_bytes=0,
                crc32=None,
                header_bits=cls.header_bits(),
                total_bits=cls.header_bits(),
            )

        total_bits = cls.header_bits() + stored_bytes * 8
        if magic != cls.MAGIC:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error="Magic mismatch",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )
        if version != cls.VERSION:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error=f"Unsupported codec version: {version}",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )
        if len(bits) < total_bits:
            return DecodedPayload(
                success=False,
                complete=False,
                text=None,
                error="Incomplete payload body",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )

        body_bits = bits[cls.header_bits():total_bits]
        body = cls.bits_to_bytes(body_bits)
        if (zlib.crc32(body) & 0xFFFFFFFF) != crc32:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error="CRC mismatch",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )

        try:
            raw = zlib.decompress(body) if (flags & cls.FLAG_COMPRESSED) else body
        except zlib.error as exc:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error=f"Decompression failed: {exc}",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )

        if len(raw) != original_bytes:
            return DecodedPayload(
                success=False,
                complete=True,
                text=None,
                error="Decoded byte length mismatch",
                compressed=bool(flags & cls.FLAG_COMPRESSED),
                original_bytes=original_bytes,
                stored_bytes=stored_bytes,
                crc32=crc32,
                header_bits=cls.header_bits(),
                total_bits=total_bits,
            )

        return DecodedPayload(
            success=True,
            complete=True,
            text=raw.decode(encoding, errors=errors),
            error=None,
            compressed=bool(flags & cls.FLAG_COMPRESSED),
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
            crc32=crc32,
            header_bits=cls.header_bits(),
            total_bits=total_bits,
        )
