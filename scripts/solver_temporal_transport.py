"""Bounded, private byte transport for trusted temporal research processes.

Hashes identify bytes only. Large snapshots remain structurally and numerically
unvalidated until a separately budgeted audit admits them. This is not a journal
or a security boundary against a malicious same-user process.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat


PROFILE = "conditional-temporal-process-transport-v1"
NAMES = ("snapshot.json", "outcome.json")


def artifact_name(name):
    """Return the fixed artifact role and whether the name is temporary."""
    if name in NAMES:
        return name, False
    if type(name) is str:
        match = re.fullmatch(r"\.(snapshot\.json|outcome\.json)\.[0-9a-f]{16}\.tmp", name)
        if match:
            return match[1], True
    raise ValueError("Unexpected temporal artifact name")


@dataclass(frozen=True)
class ExportPolicy:
    snapshot_bytes: int
    metadata_bytes: int
    log_bytes: int

    def __post_init__(self):
        for value, maximum in ((self.snapshot_bytes, 2**31), (self.metadata_bytes, 2**20),
                               (self.log_bytes, 2**26)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("Explicit bounded positive export byte limits required")

    def file_limit(self, name):
        role, _ = artifact_name(name)
        return self.snapshot_bytes if role == "snapshot.json" else self.metadata_bytes

    @property
    def file_size_limit(self):
        return max(self.snapshot_bytes, self.metadata_bytes, self.log_bytes)

    @property
    def maximum_named_bytes(self):
        # Each publication may leave a temporary name and a final hard link.
        # They share storage, but this also bounds a conservative name-wise sum.
        return 2*(self.snapshot_bytes+self.metadata_bytes)+self.log_bytes


def require_policy(policy):
    if type(policy) is not ExportPolicy:
        raise ValueError("An exact ExportPolicy is required")
    policy.__post_init__()


def require_identity(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("The declared run identity must be a SHA256 string")
    return value


def error_record(error):
    """Best-effort bounded diagnostics; formatting is not a recovery guarantee."""
    try:
        name = type(error).__name__[:128]
        try:
            message = str(error)
            incomplete = len(message) > 1024
            message = message[:1024]
        except BaseException:
            message, incomplete = "Exception message unavailable", True
        return {"type": name, "message": message, "messageIncomplete": incomplete}
    except BaseException:
        return {"type": "UnavailableError", "message": "Diagnostic allocation failed",
                "messageIncomplete": True}


def _open_directory(path):
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


class PrivateDirectory:
    """Own an anchored directory descriptor under trusted existing ancestors."""
    def __init__(self, fd, policy):
        # The constructor takes ownership even when validation fails.
        self.fd = fd
        try:
            require_policy(policy)
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("Owned private evidence directory required")
        except BaseException:
            self.close()
            raise
        self.fd, self.policy = fd, policy
        self.observations = {}
        self.attempted = set()

    @classmethod
    def create(cls, path, policy):
        require_policy(policy)
        path = Path(path)
        if path.name in ("", ".", ".."):
            raise ValueError("A new named private directory is required")
        parent = _open_directory(path.parent)
        try:
            info = os.fstat(parent)
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ValueError("The trusted parent must be owned and not writable by others")
            os.mkdir(path.name, 0o700, dir_fd=parent)
            fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
        finally:
            os.close(parent)
        return cls(fd, policy)

    @classmethod
    def open(cls, path, policy):
        require_policy(policy)
        fd = _open_directory(path)
        return cls(fd, policy)

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def publish(self, name, value):
        if name not in NAMES or name in self.attempted:
            raise ValueError("Each fixed artifact may be attempted only once")
        self.attempted.add(name)
        # Refuse an existing regular file, link, directory or FIFO before encode.
        try:
            os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(name)
        temporary = "."+name+"."+secrets.token_hex(8)+".tmp"
        row = {"path": name, "temporary": temporary, "bytes": 0, "sha256": None,
               "contentComplete": False, "fileSynced": False, "finalLinked": False,
               "temporaryRemoved": False, "directorySynced": False}
        self.observations[name] = row
        fd = None
        primary = None
        digest = hashlib.sha256()
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                         0o600, dir_fd=self.fd)
            encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=True, allow_nan=False,
                                       separators=(",", ":"), check_circular=True)
            # iterencode avoids a second whole-document string. A large scalar
            # can still allocate a large chunk; the declared process memory cap
            # remains necessary, as does memory for snapshot acquisition itself.
            def write(content):
                if row["bytes"]+len(content) > self.policy.file_limit(name):
                    raise ValueError("Temporal artifact byte budget exhausted")
                view = memoryview(content)
                while view:
                    count = os.write(fd, view[:65536])
                    if count <= 0:
                        raise OSError("Evidence write made no progress")
                    digest.update(view[:count]); row["bytes"] += count
                    view = view[count:]
            for chunk in encoder.iterencode(value):
                write(chunk.encode("utf-8"))
            write(b"\n")
            row.update(contentComplete=True, sha256=digest.hexdigest())
            os.fchmod(fd, 0o400)
            os.fsync(fd); row["fileSynced"] = True
            os.link(temporary, name, src_dir_fd=self.fd, dst_dir_fd=self.fd, follow_symlinks=False)
            row["finalLinked"] = True
            os.unlink(temporary, dir_fd=self.fd); row["temporaryRemoved"] = True
            os.fsync(self.fd); row["directorySynced"] = True
            return {"path": name, "bytes": row["bytes"], "sha256": row["sha256"]}
        except BaseException as error:
            primary = error
            row["publicationFailure"] = error_record(error)
            raise
        finally:
            # Never remove a final file after it was linked, even if a later
            # sync or cleanup fails. The parent retains any surviving bytes.
            cleanup = []
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as error:
                    cleanup.append(error)
            if not row["temporaryRemoved"]:
                try:
                    os.unlink(temporary, dir_fd=self.fd)
                    row["temporaryRemoved"] = True
                except FileNotFoundError:
                    pass
                except BaseException as error:
                    cleanup.append(error)
            if cleanup:
                row["cleanupFailures"] = [error_record(error) for error in cleanup]
                if primary is None:
                    raise cleanup[0]

    def observe(self, name, limit):
        """Stream bytes from one regular no-follow file; do not parse JSON."""
        temporary = False
        if name != "worker.log":
            _, temporary = artifact_name(name)
        if type(limit) is not int or limit < 1:
            raise ValueError("Positive byte limit required")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=self.fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise ValueError("Bounded regular temporal evidence file required")
            size, digest = 0, hashlib.sha256()
            while True:
                chunk = os.read(fd, min(65536, limit-size+1))
                if not chunk: break
                size += len(chunk)
                if size > limit: raise ValueError("Evidence grew beyond its byte budget")
                digest.update(chunk)
            after = os.fstat(fd)
            if size != before.st_size:
                raise ValueError("Evidence size does not match bytes read")
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Evidence changed while hashing")
            return {"path": name, "bytes": size, "sha256": digest.hexdigest(),
                    "temporary": temporary,
                    "validation": "bytes-only-structural-and-numerical-audit-deferred"}
        finally:
            os.close(fd)
