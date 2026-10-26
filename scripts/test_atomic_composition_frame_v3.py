#!/usr/bin/env python3
import base64
import struct
import unittest
import atomic_composition_frame_v1 as f1
import atomic_composition_frame_v2 as f2
import atomic_composition_frame_v3 as f3

class Tests(unittest.TestCase):
    def test_scope_and_all_terminals(self):
        raw=struct.pack('<'+'q'*600,*([-200,1,2,-200,3,4]*100))
        for term in [None,*sorted(f3.PRIMITIVE_TERMINALS),*sorted(f3.ALGORITHM_TERMINALS)]:
            specs=[{'op':'split_common'},{'op':'delta'},{'op':'zigzag'},{'op':'bitpack'}]
            terms=[] if term is None else [term]
            buf=f3.encode_frame(raw,'int64',specs,terms)
            self.assertEqual(f3.decode_frame(buf),raw)
            info=f3.inspect_frame(buf)
            self.assertEqual(info['format'],'ACF3')
            self.assertEqual(info['version'],3)
            self.assertEqual(info['registered_depth'],4+bool(terms))
            self.assertEqual(info['metadata'][0]['common_value'],(1<<64)-200)
            self.assertEqual(len(base64.b64decode(info['metadata'][0]['mask_b64'])),75)
            self.assertGreater(info['metadata_bytes'],75)
    def test_v1_v2_unchanged(self):
        raw=struct.pack('<q',4)
        self.assertEqual(f1.encode_frame(raw,'int64',[],[])[:4],b'ACF1')
        self.assertEqual(f2.encode_frame(raw,'int64',[],[])[:4],b'ACF2')
        with self.assertRaises(ValueError):f2.decode_frame(f3.encode_frame(raw,'int64',[],[]))
    def test_empty_all_same_and_float_bits(self):
        for values in [[],[7]*100,[0x7ff8000000000001,0x8000000000000000,0x7ff8000000000001]]:
            raw=struct.pack('<'+'Q'*len(values),*values)
            for terms in [[],['bit8'],['sparse1']]:
                buf=f3.encode_frame(raw,'float64',[{'op':'split_common'}],terms)
                self.assertEqual(f3.decode_frame(buf),raw)
    def test_depth(self):
        with self.assertRaises(ValueError):
            f3.encode_frame(struct.pack('<q',1),'int64',[{'op':'split_common'}]*10,['bit8'])

if __name__=='__main__': unittest.main()
