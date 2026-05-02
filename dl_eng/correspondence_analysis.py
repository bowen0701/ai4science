#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy as sp 
import pandas as pd
from numpy.linalg import svd


class CorrespondenceAnalysis:
    """Correspondence analysis (CA).
    
    Methods:
      fit: Fit correspondence analysis.
      get_coordinates: Get symmetric or asymmetric map coordinates.
      score_inertia: Get score inertia.

    ### Usage

    ```python
    ca = CorrespondenceAnalysis(agg_df)
    ca.fit()

    coord_df = ca.get_coordinates(option='symmetric')
    inertia_prop = ca.score_inertia()
    ```
    """

    def __init__(self, agg_df):
        """Create a new Correspondence Analysis.
        
        Args:
          agg_df: Pandas DataFrame, with row and column names.
          
        Raises:
          TypeError: Input data  is not a pandas DataFrame
          ValueError: Input data  contains missing values.
          TypeError: Input data  contains data types other than numeric.
        """
        self.agg_df = agg_df

        if isinstance(self.agg_df, pd.DataFrame) is not True:
            raise TypeError('Input data is not a Pandas DataFrame.')

        self._rows = np.array(self.agg_df.index)
        self._cols = np.array(self.agg_df.columns)
        self._np_data = np.array(self.agg_df.values)

        if np.isnan(self._np_data).any():
            raise ValueError('Input data contains missing values.')
        if np.issubdtype(self._np_data.dtype, np.number) is not True:
            raise TypeError('Input data contains data types other than numeric.')

    def fit(self):
        """Compute Correspondence Analysis.

        Fit method is to
          - perform generalized singular value decomposition (SVD) for 
            correspondence matrix and 
          - compute principal and standard coordinates for rows and columns.

        Returns:
          self: Object.
        """     
        p_corrmat = self._np_data / self._np_data.sum()
        r_profile = p_corrmat.sum(axis=1).reshape(p_corrmat.shape[0], 1)
        c_profile = p_corrmat.sum(axis=0).reshape(p_corrmat.shape[1], 1)
        Dr_invsqrt = np.diag(np.power(r_profile, -1/2).T[0])
        Dc_invsqrt = np.diag(np.power(c_profile, -1/2).T[0])
        ker_mat = np.subtract(p_corrmat, np.dot(r_profile, c_profile.T))
        left_mat = Dr_invsqrt
        right_mat = Dc_invsqrt
        weighted_lse = left_mat.dot(ker_mat).dot(right_mat)
        U, sv, Vt = svd(weighted_lse, full_matrices=False)
        self._Dr_invsqrt = Dr_invsqrt
        self._Dc_invsqrt = Dc_invsqrt
        self._U = U
        self._V = Vt.T
        self._SV = np.diag(sv)
        self._inertia = np.power(sv, 2)

        # Pricipal coordinates for rows and columns.
        self._F = self._Dr_invsqrt.dot(self._U).dot(self._SV)
        self._G = self._Dc_invsqrt.dot(self._V).dot(self._SV)

        # Standard coordinates for rows and columns.
        self._Phi = self._Dr_invsqrt.dot(self._U)
        self._Gam = self._Dc_invsqrt.dot(self._V)
        return self
    
    def _coordinates_df(self, array_x1, array_x2):
        """Create pandas DataFrame with coordinates in rows and columns.
        
        Args:
          array_x1: numpy array, coordinates in rows.
          array_x2: numpy array, coordinates in columns.
        
        Returns:
          coord_df: A Pandas DataFrame with columns 
            {'x_1',..., 'x_K', 'point', 'coord'}:
            - x_k, k=1,...,K: K-dimensional coordinates.
            - point: row and column points for labeling.
            - coord: {'row', 'col'}, indicates row point or column point.
        """
        row_df = pd.DataFrame(
            array_x1, 
            columns=['x' +  str(i) for i in (np.arange(array_x1.shape[1]) + 1)])
        row_df['point'] = self._rows
        row_df['coord'] = 'row'
        col_df = pd.DataFrame(
            array_x2, 
            columns=['x' +  str(i) for i in (np.arange(array_x2.shape[1]) + 1)])
        col_df['point'] = self._cols
        col_df['coord'] = 'col'         
        coord_df = pd.concat([row_df, col_df], ignore_index=True)
        return coord_df
    
    def get_coordinates(self, option='symmetric'):
        """Take coordinates in rows and columns for symmetric or assymetric map.
        
        For symmetric vs. asymmetric map:
          - For symmetric map, we can catch row-to-row and column-to-column 
            association (maybe not row-to-column association); 
          - For asymmetric map, we can further catch row-to-column association.
        
        Args:
          option: string, in one of the following three:
            'symmetric': symmetric map with 
              - rows and columns in principal coordinates.
            'rowprincipal': asymmetric map with 
              - rows in principal coordinates and 
              - columns in standard coordinates.  
            'colprincipal': asymmetric map with 
              - rows in standard coordinates and 
              - columns in principal coordinates.
        
        Returns:
          Pandas DataFrame, contains coordinates, row and column points.
        
        Raises:
          ValueError: Option only includes {"symmetric", "rowprincipal", "colprincipal"}.
        """     
        if option == 'symmetric':
            # Symmetric map
            return self._coordinates_df(self._F, self._G)
        elif option == 'rowprincipal':
            # Row principal asymmetric map
            return self._coordinates_df(self._F, self._Gam)
        elif option == 'colprincipal':
            # Column principal asymmetric map
            return self._coordinates_df(self._Phi, self._G)
        else:
            raise ValueError(
                'Option only includes {"symmetric", "rowprincipal", "colprincipal"}.')

    def score_inertia(self):
        """Score inertia.
        
        Returns:
          A NumPy array, contains proportions of total inertia explained 
            in coordinate dimensions.
        """
        inertia = self._inertia
        inertia_prop = (inertia / inertia.sum()).cumsum()
        return inertia_prop
