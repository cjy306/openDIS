"""Offline behavioral checks: python -B -m unittest check_scripts -v."""
import unittest
import numpy as np

class Checks(unittest.TestCase):
    def test_layout(self):
        from generate import make_layout
        a = make_layout(12345)
        np.testing.assert_array_equal(a['centers'], make_layout(12345)['centers'])
        self.assertEqual(a['centers'].shape, (250, 3))
        self.assertEqual(sorted(np.bincount(a['systems']).tolist()), [20]*2+[21]*10)
        self.assertTrue(np.allclose(np.sum(a['burgers']*a['planes'], axis=1), 0))
        self.assertTrue(np.allclose(np.linalg.norm(a['burgers'], axis=1), 1))
        self.assertEqual(a['loop_centers'].shape, (1000, 3))
        self.assertEqual(np.bincount(a['loop_types']).tolist(), [250]*4)
        self.assertFalse(np.array_equal(a['centers'], make_layout(7)['centers']))

    def test_window(self):
        from relaxation import WindowMonitor
        m = WindowMonitor()
        self.assertIsNone(m.add(5e-9, [2e-7,0,0,0,0,0]))
        result = m.add(5e-9, [-2e-7,0,0,0,0,0])
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['activity'], 4e-7)
        self.assertIsNone(m.add(5e-9, [0,0,0,0,0,0]))
        self.assertFalse(m.add(5e-9, [0,0,0,1e-7,0,0])['converged'])

    def test_real_geometry(self):
        import ast
        from types import SimpleNamespace
        from runtime import setup
        from generate import make_layout, build_arrays
        source = setup()/'core/exadis/python/pyexadis_utils.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and
                     n.name in ('insert_frank_read_src','insert_prismatic_loop')]
        scope = dict(np=np,NodeConstraints=SimpleNamespace(PINNED_NODE=7,UNCONSTRAINED=0))
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),scope)
        cell = SimpleNamespace(origin=np.zeros(3),h=np.eye(3)*(1e-6/0.248e-9))
        args = (make_layout(),cell,scope['insert_frank_read_src'],scope['insert_prismatic_loop'])
        a,sa,nn,ns = build_arrays(*args,False)
        b,sb,_,_ = build_arrays(*args,True)
        np.testing.assert_array_equal(a,b[:nn])
        np.testing.assert_array_equal(sa,sb[:ns])
        self.assertEqual((nn,ns),(20500,20250))
        self.assertEqual(len(b)-nn,18000)
        ends=sb[:,:2].astype(int)
        delta=b[ends[:,1],:3]-b[ends[:,0],:3]
        lengths=np.linalg.norm(delta,axis=1)*0.248e-9
        self.assertAlmostEqual(lengths[:ns].sum()/1e-4,1,places=10)
        self.assertAlmostEqual(lengths[ns:].sum()/6e-5,1,places=10)
        self.assertTrue(np.all(np.linalg.norm(delta,axis=1)<=20+1e-9))
        self.assertTrue(np.allclose(np.sum(delta*sb[:,5:8],axis=1),0,atol=1e-9))
        self.assertTrue(np.all(b[nn:,3]==0))

    def test_plot(self):
        from plot import parse_response
        e,s,r = parse_response('# Step Strain Stress Density\n0 0 0 1e14\n1 0.005 2e8 2e14\n')
        self.assertEqual(e[-1], 0.5)
        self.assertEqual(s[-1], 200)
        self.assertEqual(r[-1], 2e14)
        with self.assertRaises(ValueError):
            parse_response('0 0.005 2e8 1e14\n1 0.004 2e8 1e14\n')

if __name__ == '__main__':
    unittest.main()
