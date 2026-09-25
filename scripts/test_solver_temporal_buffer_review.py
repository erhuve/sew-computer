"""Independent byte/publication contracts for bounded temporal JSON buffering.

These tests use real temporary files and canonical stdlib JSON as the oracle.
They do not exercise mechanics or make performance/timing assertions.
"""
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import solver_temporal_transport as transport


BLOCK = 65536


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('ascii')


class TemporalBufferIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def store(self, name='evidence', cap=2**21, metadata=2**20):
        return transport.PrivateDirectory.create(
            self.root/name, transport.ExportPolicy(cap, metadata, 4096))

    def assert_final(self, folder, row, expected, name='snapshot.json'):
        self.assertEqual((folder/name).read_bytes(), expected)
        self.assertEqual(row['bytes'], len(expected))
        self.assertEqual(row['sha256'], hashlib.sha256(expected).hexdigest())
        self.assertEqual(sorted(p.name for p in folder.iterdir()), [name])

    def capture_unlink(self, folder, captures):
        original = os.unlink
        def unlink(name, *args, **kwargs):
            if str(name).startswith('.snapshot.json.'):
                captures.append((folder/name).read_bytes())
            return original(name, *args, **kwargs)
        return unlink

    def test_canonical_unicode_signed_zero_large_scalar_and_nested_records(self):
        cases = [
            {'z': [None, True, False, -0.0, 0.0, 2**-1074],
             'a': 'Ω😀\n\t\x00\\"\ud800'},
            {'scalar': 'x'*(3*BLOCK+117) + 'Ω😀'},
            {'rows': [{'id': i, 'values': [i/8, -0.0, None], 'text': 'δ'*13}
                      for i in range(1600)], 'tail': {'fraction': {'numerator': '1', 'denominator': '3'}}},
        ]
        for index, value in enumerate(cases):
            with self.subTest(case=index), self.store(str(index)) as store:
                expected = canonical(value)
                row = store.publish('snapshot.json', value)
                self.assert_final(self.root/str(index), row, expected)

    def test_many_small_tokens_coalesce_and_every_write_request_is_bounded(self):
        value = {'rows': [[i, i/16, 'xy', None, True] for i in range(9000)]}
        expected, requests = canonical(value), []
        original = os.write
        def observe(fd, data):
            requests.append(len(data))
            return original(fd, data)
        with self.store() as store, patch.object(transport.os, 'write', observe):
            row = store.publish('snapshot.json', value)
        self.assert_final(self.root/'evidence', row, expected)
        self.assertGreater(len(requests), 3)
        self.assertTrue(all(0 < size <= BLOCK for size in requests))
        self.assertLessEqual(len(requests), math.ceil(len(expected)/BLOCK)+1)
        self.assertEqual(sum(requests), len(expected))

    def test_short_writes_preserve_complete_canonical_bytes_and_digest(self):
        value = {'a': 'abcΩ'*(BLOCK//2), 'z': list(range(17))}
        expected, requests, returned = canonical(value), [], []
        original = os.write
        def short(fd, data):
            requests.append(len(data))
            allowance = (1, 19, 4093, 10007)[len(requests) % 4]
            count = original(fd, data[:allowance])
            returned.append(count)
            return count
        with self.store() as store, patch.object(transport.os, 'write', short):
            row = store.publish('snapshot.json', value)
        self.assert_final(self.root/'evidence', row, expected)
        self.assertTrue(all(0 < size <= BLOCK for size in requests))
        self.assertEqual(sum(returned), len(expected))
        self.assertGreater(len(requests), math.ceil(len(expected)/BLOCK))

    def test_exact_cap_includes_newline_across_buffer_boundaries(self):
        for target in (BLOCK-1, BLOCK, BLOCK+1, 2*BLOCK+1):
            # A JSON string plus newline contributes three framing bytes.
            value = 'a'*(target-3)
            expected = canonical(value)
            self.assertEqual(len(expected), target)
            with self.subTest(target=target), self.store(str(target), cap=target) as store:
                row = store.publish('snapshot.json', value)
                self.assert_final(self.root/str(target), row, expected)

    def test_one_byte_over_cap_never_links_or_counts_unwritten_pending_bytes(self):
        for target in (BLOCK-1, BLOCK, BLOCK+1, 2*BLOCK+1):
            value, captures = 'b'*(target-3), []
            expected = canonical(value)
            folder = self.root/str(target)
            with self.subTest(target=target), self.store(str(target), cap=target-1) as store:
                with patch.object(transport.os, 'unlink', self.capture_unlink(folder, captures)):
                    with self.assertRaises(ValueError): store.publish('snapshot.json', value)
                row = store.observations['snapshot.json']
                persisted = captures[0]
                self.assertEqual(row['bytes'], len(persisted))
                self.assertLessEqual(len(persisted), target-1)
                self.assertEqual(persisted, expected[:len(persisted)])
                self.assertFalse(row['contentComplete'])
                self.assertFalse(row['finalLinked'])
                self.assertIsNone(row['sha256'])
                self.assertEqual(list(folder.iterdir()), [])

    def test_partial_write_then_enospc_retains_actual_prefix_accounting(self):
        value, captures = {'data': 'x'*(3*BLOCK+97)}, []
        expected, calls, counts = canonical(value), [], []
        folder, original = self.root/'evidence', os.write
        def fail(fd, data):
            calls.append(len(data))
            if len(calls) == 3:
                raise OSError(errno.ENOSPC, 'declared disk-full after short write')
            count = original(fd, data if len(calls) == 1 else data[:17])
            counts.append(count)
            return count
        with self.store() as store:
            with patch.object(transport.os, 'write', fail), \
                 patch.object(transport.os, 'unlink', self.capture_unlink(folder, captures)):
                with self.assertRaises(OSError) as caught: store.publish('snapshot.json', value)
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            row = store.observations['snapshot.json']
            self.assertEqual(row['bytes'], sum(counts))
            self.assertEqual(captures, [expected[:sum(counts)]])
            self.assertTrue(all(0 < size <= BLOCK for size in calls))
            self.assertFalse(row['contentComplete'])
            self.assertFalse(row['finalLinked'])
            self.assertTrue(row['temporaryRemoved'])
            self.assertIsNone(row['sha256'])
            self.assertEqual(list(folder.iterdir()), [])

    def test_zero_progress_does_not_publish_or_count_pending_data(self):
        captures, folder = [], self.root/'evidence'
        with self.store() as store:
            with patch.object(transport.os, 'write', return_value=0), \
                 patch.object(transport.os, 'unlink', self.capture_unlink(folder, captures)):
                with self.assertRaises(OSError): store.publish('snapshot.json', {'x':'x'*(2*BLOCK)})
            row = store.observations['snapshot.json']
            self.assertEqual(row['bytes'], 0)
            self.assertEqual(captures, [b''])
            self.assertFalse(row['finalLinked'])
            self.assertEqual(list(folder.iterdir()), [])

    def test_encoding_failure_never_flushes_pending_buffer_during_cleanup(self):
        for index, text in enumerate(('tiny', 'large'*(BLOCK//2))):
            captures, requests = [], []
            folder, original = self.root/str(index), os.write
            def observe(fd, data):
                requests.append(len(data)); return original(fd, data)
            # Sorted key z fails after a complete valid prefix has been encoded.
            prefix = ('{"a":'+json.dumps(text)+',"z":').encode('ascii')
            with self.subTest(case=index), self.store(str(index)) as store:
                with patch.object(transport.os, 'write', observe), \
                     patch.object(transport.os, 'unlink', self.capture_unlink(folder, captures)):
                    with self.assertRaises(ValueError):
                        store.publish('snapshot.json', {'a':text, 'z':float('nan')})
                row = store.observations['snapshot.json']
                self.assertEqual(captures[0], prefix[:row['bytes']])
                self.assertEqual(row['bytes'], len(captures[0]))
                self.assertFalse(row['contentComplete'])
                self.assertFalse(row['finalLinked'])
                self.assertEqual(list(folder.iterdir()), [])
                if index == 0:
                    self.assertEqual(requests, [])
                    self.assertEqual(row['bytes'], 0)

    def test_interruption_after_returned_short_write_keeps_primary_and_prefix(self):
        value, captures = {'x':'q'*(2*BLOCK)}, []
        expected, original, calls = canonical(value), os.write, []
        primary = KeyboardInterrupt('declared buffer flush interruption')
        folder = self.root/'evidence'
        def interrupted(fd, data):
            calls.append(len(data))
            if len(calls) == 2: raise primary
            return original(fd, data[:23])
        with self.store() as store:
            with patch.object(transport.os, 'write', interrupted), \
                 patch.object(transport.os, 'unlink', self.capture_unlink(folder, captures)):
                with self.assertRaises(KeyboardInterrupt) as caught: store.publish('snapshot.json', value)
            self.assertIs(caught.exception, primary)
            self.assertEqual(captures, [expected[:23]])
            self.assertEqual(store.observations['snapshot.json']['bytes'], 23)
            self.assertFalse(store.observations['snapshot.json']['finalLinked'])
            self.assertEqual(list(folder.iterdir()), [])

    def test_pre_link_error_happens_only_after_complete_flushed_file(self):
        value = {'x':'link'*(BLOCK//2)}
        expected, observed = canonical(value), []
        folder = self.root/'evidence'
        def fail(source, destination, **kwargs):
            observed.append((folder/source).read_bytes())
            raise OSError(errno.EIO, 'declared pre-link failure')
        with self.store() as store, patch.object(transport.os, 'link', fail):
            with self.assertRaises(OSError): store.publish('snapshot.json', value)
            row = store.observations['snapshot.json']
            self.assertEqual(observed, [expected])
            self.assertEqual(row['bytes'], len(expected))
            self.assertEqual(row['sha256'], hashlib.sha256(expected).hexdigest())
            self.assertTrue(row['fileSynced'])
            self.assertFalse(row['finalLinked'])
            self.assertEqual(list(folder.iterdir()), [])

    def test_eexist_race_preserves_rival_after_own_buffer_flushed(self):
        folder, rival = self.root/'evidence', b'{"rival":"unchanged"}\n'
        original_link, original_write = os.link, os.write
        def race(source, destination, **kwargs):
            fd = os.open(destination, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o400,
                         dir_fd=kwargs['dst_dir_fd'])
            try: self.assertEqual(original_write(fd, rival), len(rival))
            finally: os.close(fd)
            return original_link(source, destination, **kwargs)
        with self.store() as store, patch.object(transport.os, 'link', race):
            with self.assertRaises(FileExistsError): store.publish('snapshot.json', {'own':'z'*(2*BLOCK)})
            self.assertEqual((folder/'snapshot.json').read_bytes(), rival)
            self.assertFalse(store.observations['snapshot.json']['finalLinked'])
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ['snapshot.json'])

    def test_persistent_post_link_unlink_failure_preserves_both_names(self):
        value, expected = {'x':'u'*(2*BLOCK)}, canonical({'x':'u'*(2*BLOCK)})
        folder, original, calls = self.root/'evidence', os.unlink, []
        def fail(name, *args, **kwargs):
            calls.append(str(name))
            if str(name).startswith('.snapshot.json.'):
                raise OSError(errno.EIO, 'declared unlink failure')
            return original(name, *args, **kwargs)
        with self.store() as store, patch.object(transport.os, 'unlink', fail):
            with self.assertRaises(OSError): store.publish('snapshot.json', value)
            row = store.observations['snapshot.json']
            final, temporary = folder/'snapshot.json', folder/row['temporary']
            self.assertEqual(final.read_bytes(), expected)
            self.assertEqual(temporary.read_bytes(), expected)
            self.assertEqual(final.stat().st_ino, temporary.stat().st_ino)
            self.assertTrue(row['contentComplete'])
            self.assertTrue(row['finalLinked'])
            self.assertFalse(row['temporaryRemoved'])
            self.assertFalse(row['directorySynced'])
            self.assertTrue(row['cleanupFailures'])
            self.assertNotIn('snapshot.json', calls)

    def test_directory_sync_failure_keeps_full_final_and_no_temporary(self):
        value, original = {'x':'sync'*(BLOCK//2)}, os.fsync
        expected, folder = canonical(value), self.root/'evidence'
        with self.store() as store:
            def fail(fd):
                if fd == store.fd: raise OSError(errno.EIO, 'declared directory-sync failure')
                return original(fd)
            with patch.object(transport.os, 'fsync', fail):
                with self.assertRaises(OSError): store.publish('snapshot.json', value)
            row = store.observations['snapshot.json']
            self.assert_final(folder, row, expected)
            self.assertTrue(row['fileSynced'])
            self.assertTrue(row['finalLinked'])
            self.assertFalse(row['directorySynced'])

    def test_failed_metadata_buffer_cannot_change_published_snapshot(self):
        raw, expected = {'x':'raw'*(BLOCK//2)}, canonical({'x':'raw'*(BLOCK//2)})
        folder = self.root/'evidence'
        with self.store(metadata=1) as store:
            row = store.publish('snapshot.json', raw)
            with self.assertRaises(ValueError): store.publish('outcome.json', {'complete':True})
            self.assert_final(folder, row, expected)
            failed = store.observations['outcome.json']
            self.assertEqual(failed['bytes'], 0)
            self.assertFalse(failed['finalLinked'])


if __name__ == '__main__':
    unittest.main()
