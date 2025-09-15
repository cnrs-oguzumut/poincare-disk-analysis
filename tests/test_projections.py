#!/usr/bin/env python3
"""
Unit tests for projection functions.
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.projections import (
    Cij_from_stereographic_projection,
    Cij_from_stereographic_projection_tr,
    stereographic_projection_from_Cij_2D
)

class TestProjections(unittest.TestCase):
    """Test cases for projection functions."""
    
    def setUp(self):
        """Set up test data."""
        self.test_x = np.array([0.0, 0.5, -0.3])
        self.test_y = np.array([0.0, 0.2, 0.7])
    
    def test_stereographic_projection_basic(self):
        """Test basic stereographic projection."""
        c11, c22, c12 = Cij_from_stereographic_projection(self.test_x, self.test_y)
        
        # Check that we get finite results
        self.assertTrue(np.all(np.isfinite(c11)))
        self.assertTrue(np.all(np.isfinite(c22)))
        self.assertTrue(np.all(np.isfinite(c12)))
        
        # Check positive definiteness (c11 > 0, det > 0)
        det = c11 * c22 - c12**2
        self.assertTrue(np.all(c11 > 0))
        self.assertTrue(np.all(det > 0))
    
    def test_inverse_projection(self):
        """Test that inverse projection recovers original coordinates."""
        c11, c22, c12 = Cij_from_stereographic_projection(self.test_x, self.test_y)
        x_recovered, y_recovered = stereographic_projection_from_Cij_2D(c11, c22, c12)
        
        np.testing.assert_allclose(x_recovered, self.test_x, rtol=1e-10)
        np.testing.assert_allclose(y_recovered, self.test_y, rtol=1e-10)
    
    def test_centered_projection_format(self):
        """Test that centered projection returns correct number of outputs."""
        result = Cij_from_stereographic_projection_tr(self.test_x, self.test_y)
        
        # Should return 6 arrays: c11, c22, c12, c11t, c22t, c12t
        self.assertEqual(len(result), 6)
        
        c11, c22, c12, c11t, c22t, c12t = result
        
        # All should be numpy arrays of same shape
        for arr in result:
            self.assertEqual(arr.shape, self.test_x.shape)
            self.assertTrue(np.all(np.isfinite(arr)))

if __name__ == '__main__':
    unittest.main()