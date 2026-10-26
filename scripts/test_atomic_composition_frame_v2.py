import hashlib
import unittest
import numpy as np
import atomic_composition_frame_v1 as old
import atomic_composition_frame_v2 as new
import atomic_transform_extensions_v2 as tr


class ACF2Tests(unittest.TestCase):
    def test_no_v1_mutation(self):
        raw=np.arange(32,dtype='<u8').tobytes()
        self.assertEqual(old.encode_frame(raw,'int64',[],[])[:4],b'ACF1')
        self.assertEqual(new.encode_frame(raw,'int64',[],[])[:4],b'ACF2')
        self.assertNotIn('bit4',old.TERMINALS)
        self.assertEqual(old.transforms.VERSION,'atomic-transform-v1')

    def test_blocks_all_terminals(self):
        raw=np.random.default_rng(8).integers(0,4000,257,dtype=np.uint64).astype('<u8').tobytes()
        for terms in [[],['range0'],['sparse1'],['zstd'],['bzip2','zlib6']]:
            for block in [16,128,1024]:
                specs=[{'op':'block_for','block_size':block},{'op':'block_bitpack','block_size':block}]
                f=new.encode_frame(raw,'int64',specs,terms)
                self.assertEqual(new.decode_frame(f),raw)
                self.assertEqual(new.inspect_frame(f)['registered_depth'],2+len(terms))
                self.assertEqual(new.inspect_frame(f)['format'],'ACF2')
                self.assertEqual(new.encode_prepared(raw,'int64',specs,terms,*tr.apply_ops(raw,specs)),f)

    def test_float_exception_side_streams_charged(self):
        a=np.arange(100,dtype='<f8')/10
        a.view('<u8')[4]=0x7ff8000000000011
        a[70]=-0.
        specs=[{'op':'float_scale_exceptions','factor':10}, {'op':'delta'}, {'op':'zigzag'}, {'op':'block_bitpack','block_size':32}]
        raw=a.tobytes()
        for terms in [[],['range1'],['xz']]:
            f=new.encode_frame(raw,'float64',specs,terms)
            self.assertEqual(new.decode_frame(f),raw)
            i=new.inspect_frame(f)
            self.assertGreater(i['metadata_bytes'],32)
            self.assertEqual(i['uncompressed_body_bytes'],i['metadata_bytes']+i['metadata_length_prefix_bytes']+i['payload_bytes'])

    def test_prep_and_depth_reject(self):
        raw=np.arange(100,dtype='<u8').tobytes()
        specs=[{'op':'block_bitpack','block_size':16}]
        payload,meta=tr.apply_ops(raw,specs)
        with self.assertRaises(ValueError):new.encode_prepared(raw,'int64',[{'op':'block_bitpack','block_size':128}],[],payload,meta)
        with self.assertRaises(ValueError):new.encode_frame(raw,'int64',[{'op':'gcd'}]*10,['range0'])

    def test_damage_and_wrong_version(self):
        raw=np.arange(100,dtype='<u8').tobytes()
        f=new.encode_frame(raw,'int64',[{'op':'block_bitpack'}],['sparse1'])
        for damaged in [f[:-1],f+b'junk',f[:20]+bytes([f[20]^1])+f[21:]]:
            with self.assertRaises(ValueError):new.decode_frame(damaged)
        with self.assertRaises(ValueError):old.decode_frame(f)
        with self.assertRaises(ValueError):new.decode_frame(old.encode_frame(raw,'int64',[],[]))

    def test_binary_modes(self):
        if not new.BINARY_CLI.is_file():self.skipTest('binary entropy v2 not built yet')
        raw=np.arange(512,dtype='<u8').tobytes()
        for term in new.BIT_TERMINALS:
            f=new.encode_frame(raw,'int64',[{'op':'delta'},{'op':'zigzag'},{'op':'compact'}],[term])
            self.assertEqual(new.decode_frame(f),raw)
            self.assertEqual(new.inspect_frame(f)['terminals'],[term])


if __name__=='__main__':unittest.main(verbosity=2)
