import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path


def add_source_tree_package() -> bool:
    source_tree = os.getenv("QEMU_HEDGEHOG_SOURCE_TREE")
    if not source_tree:
        return False

    package_root = Path(source_tree).resolve() / "python"
    if not package_root.is_dir():
        return False

    package_root_str = str(package_root)
    if package_root_str not in sys.path:
        sys.path.insert(0, package_root_str)
    return True


def add_local_site_packages() -> None:
    if add_source_tree_package():
        return

    root = Path(__file__).resolve().parent
    for env_name in ("venv", ".venv"):
        lib_dir = root / env_name / "lib"
        if not lib_dir.is_dir():
            continue
        for site_packages in sorted(lib_dir.glob("python*/site-packages")):
            if site_packages.is_dir() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
                return


add_local_site_packages()

from qemu.hedgehog import (
    Hedgehog,
    HEDGEHOG_ARCH_ARM64,
    HEDGEHOG_HOOK_CODE,
    HEDGEHOG_MODE_ARM,
)
from qemu.hedgehog.errors import HedgehogError


ELF_MAGIC = b"\x7fELF"
ELFCLASS64 = 2
ELFDATA2LSB = 1
EM_AARCH64 = 183
PT_LOAD = 1
PAGE_SIZE = 0x1000


@dataclass(frozen=True)
class LoadSegment:
    offset: int
    vaddr: int
    filesz: int
    memsz: int


def align_down(value: int, alignment: int) -> int:
    return value & ~(alignment - 1)


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def parse_elf64_aarch64(path: Path) -> tuple[int, list[LoadSegment]]:
    with path.open("rb") as f:
        ehdr = f.read(64)
        if len(ehdr) != 64:
            raise ValueError("ELF header is truncated")
        if ehdr[:4] != ELF_MAGIC:
            raise ValueError("Not an ELF file")
        if ehdr[4] != ELFCLASS64:
            raise ValueError("Only ELF64 is supported")
        if ehdr[5] != ELFDATA2LSB:
            raise ValueError("Only little-endian ELF is supported")

        (
            _ident,
            _etype,
            machine,
            _version,
            entry,
            phoff,
            _shoff,
            _flags,
            _ehsize,
            phentsize,
            phnum,
            _shentsize,
            _shnum,
            _shstrndx,
        ) = struct.unpack("<16sHHIQQQIHHHHHH", ehdr)

        if machine != EM_AARCH64:
            raise ValueError(f"ELF machine is not AArch64 (got {machine})")

        segments: list[LoadSegment] = []
        for idx in range(phnum):
            f.seek(phoff + idx * phentsize)
            phdr = f.read(phentsize)
            if len(phdr) != phentsize:
                raise ValueError("Program header table is truncated")

            p_type, _p_flags, p_offset, p_vaddr, _p_paddr, p_filesz, p_memsz, _p_align = struct.unpack(
                "<IIQQQQQQ", phdr[:56]
            )
            if p_type == PT_LOAD and p_memsz > 0:
                segments.append(
                    LoadSegment(
                        offset=p_offset,
                        vaddr=p_vaddr,
                        filesz=p_filesz,
                        memsz=p_memsz,
                    )
                )

        if not segments:
            raise ValueError("ELF has no PT_LOAD segments")

        segments.sort(key=lambda seg: seg.vaddr)
        return entry, segments


def merged_load_ranges(segments: list[LoadSegment]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for seg in segments:
        start = align_down(seg.vaddr, PAGE_SIZE)
        end = align_up(seg.vaddr + seg.memsz, PAGE_SIZE)
        ranges.append((start, end))

    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return [(start, end) for start, end in merged]


def write_zero_fill(emu: Hedgehog, address: int, size: int, chunk_size: int = 0x10000) -> None:
    zero_chunk = b"\x00" * min(chunk_size, size)
    written = 0
    while written < size:
        step = min(len(zero_chunk), size - written)
        emu.mem_write(address + written, zero_chunk[:step])
        written += step


def load_elf_into_emulator(emu: Hedgehog, elf_path: Path) -> int:
    entry, segments = parse_elf64_aarch64(elf_path)

    for start, end in merged_load_ranges(segments):
        try:
            emu.mem_map(start, end - start)
        except HedgehogError:
            # Board machine models may already have RAM in this address range.
            pass

    with elf_path.open("rb") as f:
        for seg in segments:
            f.seek(seg.offset)
            payload = f.read(seg.filesz)
            if len(payload) != seg.filesz:
                raise ValueError("ELF segment payload is truncated")

            emu.mem_write(seg.vaddr, payload)
            if seg.memsz > seg.filesz:
                write_zero_fill(emu, seg.vaddr + seg.filesz, seg.memsz - seg.filesz)

    return entry


def main() -> None:
    elf_path = Path(__file__).with_name("kernel8.elf")
    if not elf_path.exists():
        raise FileNotFoundError(f"Missing ELF file: {elf_path}")

    instruction_budget = int(os.getenv("HEDGEHOG_MAX_INSN", "200000000"))
    run_chunk = int(os.getenv("HEDGEHOG_RUN_CHUNK", "10000"))
    trace_pc = os.getenv("HEDGEHOG_TRACE_PC", "0") == "1"

    with Hedgehog(
        HEDGEHOG_ARCH_ARM64,
        HEDGEHOG_MODE_ARM,
        cpu_type="cortex-a53",
        machine_type="raspi3b",
        chardevs={"console": "pty"},
        property_bindings={
            "/machine/soc/peripherals/uart0": {
                "chardev": "console",
            },
        },
    ) as emu:
        if trace_pc:
            def log_pc_hook(_emu: Hedgehog, pc: int, _size: int, _user_data: object) -> bool:
                print(f"PC=0x{pc:x}")
                return False

            emu.hook_add(HEDGEHOG_HOOK_CODE, log_pc_hook)

        entry = load_elf_into_emulator(emu, elf_path)
        emu.qemu_set_pc(entry)
        print(f"UART PTY: {emu.qemu_chardev_get_endpoint('console')}")
        print(f"Loaded {elf_path.name}, entry=0x{entry:x}")

        remaining = instruction_budget
        while remaining > 0:
            step = min(run_chunk, remaining)
            emu.emu_start(begin=emu.qemu_get_pc(), until=0, count=step)
            emu.qemu_events_poll()
            remaining -= step


if __name__ == "__main__":
    main()