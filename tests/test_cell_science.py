import sys
from pathlib import Path
import unittest
import numpy as np
from scipy import sparse
from src.paper.cells.emt_cells_v1.common import normalize,coexpression_flags,neighborhood_counts
from src.paper.cells.emt_cells_v1.prepare_cells import source_group

class ScientificChecks(unittest.TestCase):
    def test_mixed_neighbor_cells_do_not_become_same_cell_coexpression(self):
        counts=sparse.csr_matrix([[4,0,0],[0,3,2],[4,3,2]])
        flags,_,_=coexpression_flags(counts,np.array(['EPCAM','SNAI1','ZEB1']))
        np.testing.assert_array_equal(flags,[False,False,True])
    def test_one_tf_is_not_two_tf_support(self):
        flags,_,_=coexpression_flags(sparse.csr_matrix([[2,100,0]]),np.array(['EPCAM','ZEB1','ZEB2']))
        self.assertFalse(flags[0])
    def test_source_transitional_label_is_not_assumed_malignant_emt(self):
        self.assertEqual(source_group('Transitional Cells'),'Published transitional')
        self.assertEqual(source_group('T_Cell_&_Tumor_Hybrid'),'Uncertain/hybrid')
        self.assertEqual(source_group('Myoepi_ACTA2+'),'Myoepithelial')
    def test_neighborhood_denominator_only_includes_tumor(self):
        xy=np.array([[0.,0.],[1.,0.],[2.,0.],[100.,0.]])
        total,pos,fraction=neighborhood_counts(xy,[1,1,0,1],[1,0,1,1],3)
        np.testing.assert_array_equal(total,[2,2,2,1]);np.testing.assert_array_equal(pos,[1,1,1,1])
        np.testing.assert_array_equal(fraction,[.5,.5,.5,1.])
    def test_normalization_preserves_zero_and_library_scaling(self):
        x=normalize(sparse.csr_matrix([[1,3],[2,6],[0,0]])).toarray()
        np.testing.assert_array_equal(x[0],x[1]);np.testing.assert_array_equal(x[2],0)
    def test_empty_tumor_neighborhood_does_not_create_zone(self):
        total,pos,fraction=neighborhood_counts(np.array([[0.,0.],[1.,0.]]),[0,0],[1,1],10)
        self.assertFalse(total.any());self.assertFalse(pos.any());self.assertTrue(np.isnan(fraction).all())

if __name__=='__main__':unittest.main()
