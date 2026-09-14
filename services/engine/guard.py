import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import socket

ROOT = Path(__file__).resolve().parent
COMMIT = "7065b3ef01ff61f4462e871d4cb439a0b97c48db"


def verify_runtime(upstream):
    lock = json.loads((ROOT / "source-lock.json").read_text())
    head = (upstream / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        ref = head[5:]
        target = upstream / ".git" / ref
        if target.exists():
            head = target.read_text().strip()
        else:
            refs = (upstream / ".git/packed-refs").read_text().splitlines()
            head = next((line.split()[0] for line in refs if line.endswith(" " + ref)), "")
    if head != COMMIT or lock["commit"] != COMMIT:
        raise RuntimeError("Upstream source commit does not match the engine pin")
    for relative, digest in lock["sha256"].items():
        path = upstream / relative
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Upstream source/config integrity failed: " + relative)
    for line in (ROOT / "requirements.lock").read_text().splitlines():
        name, version = line.split("==")
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Pinned dependency mismatch: " + name)


def deny_network():
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
        lib.seccomp_init.argtypes = [ctypes.c_uint32]
        lib.seccomp_init.restype = ctypes.c_void_p
        lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
        lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
        lib.seccomp_rule_add.restype = ctypes.c_int
        lib.seccomp_load.argtypes = [ctypes.c_void_p]
        lib.seccomp_load.restype = ctypes.c_int
        lib.seccomp_release.argtypes = [ctypes.c_void_p]
        ctx = lib.seccomp_init(0x7FFF0000)
        if not ctx:
            raise RuntimeError("seccomp_init failed")
        try:
            for name in ("socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg", "socketcall", "io_uring_setup", "io_uring_enter", "io_uring_register"):
                number = lib.seccomp_syscall_resolve_name(name.encode())
                if number >= 0 and lib.seccomp_rule_add(ctx, 0x00050000 | errno.EPERM, number, 0) != 0:
                    raise RuntimeError("seccomp rule failed")
            if lib.seccomp_load(ctx) != 0:
                raise RuntimeError("seccomp_load unavailable")
        finally:
            lib.seccomp_release(ctx)
    except (OSError, RuntimeError):
        return "not isolated (seccomp unavailable); no network sandbox"
    for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
        try:
            with socket.socket(family, socket.SOCK_STREAM):
                pass
        except OSError as error:
            if error.errno == errno.EPERM:
                continue
            raise RuntimeError("Unexpected network probe failure") from error
        raise RuntimeError("Network-denial verification failed")
    return "seccomp socket/network syscalls denied; IPv4/IPv6/Unix socket probes returned EPERM (not a network namespace)"


def constrain():
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    if os.getuid() == 0 or 0 in os.getgroups():
        raise RuntimeError("Worker requires non-root uid and no privileged supplementary groups")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise RuntimeError("Cannot enforce no_new_privs")
    libc.prctl(4, 0, 0, 0, 0)
    return {"uid": os.getuid(), "groups": os.getgroups(), "network": deny_network(), "limits": {"cpuSeconds": 60, "addressSpaceBytes": 2 * 1024**3, "fileBytes": 16 * 1024**2, "descriptors": 64, "processes": 64}}
