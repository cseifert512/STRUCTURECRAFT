"""
ML Pipeline Smoke Tests
========================

Quick sanity tests for the Day 4 ML pipeline.
These tests verify that the ML components work without errors,
not that they produce optimal results.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMLUtils:
    """Tests for the _ml_utils helper module."""
    
    def test_load_clean_df_filters_ok(self, tmp_path):
        """Test that load_clean_df filters to ok=True only."""
        # Create a small test CSV
        csv_path = tmp_path / "test_results.csv"
        df = pd.DataFrame({
            'span': [5.0, 6.0, 7.0],
            'height': [3.0, 3.5, 4.0],
            'brace': [0, 1, 0],
            'sec_col': [2, 3, 4],
            'sec_beam': [2, 3, 4],
            'sec_brace': [1, 2, 3],
            'udl_w': [-2000, -2500, -3000],
            'wind_P': [5000, 6000, 7000],
            'shipping_limit': [12.0, 12.0, 12.0],
            'drift': [0.1, 0.2, np.nan],  # Third row has NaN
            'max_abs_M': [1000, 2000, 3000],
            'volume': [0.1, 0.2, 0.3],
            'carbon': [30, 60, 90],
            'ok': [True, True, False],  # Third row is not ok
            'reason': ['', '', 'failed'],
        })
        df.to_csv(csv_path, index=False)
        
        # Import and test
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demos'))
        from _ml_utils import load_clean_df
        
        result = load_clean_df(str(csv_path))
        
        # Should only have 2 rows (ok=True and no NaN)
        assert len(result) == 2
        assert all(result['ok'] == True)
    
    def test_make_features_shape(self):
        """Test that make_features returns correct shape."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demos'))
        from _ml_utils import make_features
        
        df = pd.DataFrame({
            'span': [5.0, 6.0, 7.0],
            'height': [3.0, 3.5, 4.0],
            'brace': [0, 1, 0],
            'sec_col': [2, 3, 4],
            'sec_beam': [2, 3, 4],
            'sec_brace': [1, 2, 3],
            'udl_w': [-2000, -2500, -3000],
            'wind_P': [5000, 6000, 7000],
        })
        
        X, feature_names = make_features(df)
        
        assert X.shape == (3, 8)
        assert len(feature_names) == 8
        assert 'span' in feature_names
        assert 'brace' in feature_names
    
    def test_make_target_shape(self):
        """Test that make_target returns correct shape."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demos'))
        from _ml_utils import make_target
        
        df = pd.DataFrame({
            'drift': [0.1, 0.2, 0.3],
            'volume': [0.1, 0.2, 0.3],
        })
        
        y = make_target(df, target='drift')
        
        assert y.shape == (3,)
        assert y[0] == 0.1
    
    def test_params_to_features(self):
        """Test converting parameter dicts to feature matrix."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'demos'))
        from _ml_utils import params_to_features
        
        params_list = [
            {'span': 5.0, 'height': 3.0, 'brace': 0, 'sec_col': 2, 
             'sec_beam': 2, 'sec_brace': 1, 'udl_w': -2000, 'wind_P': 5000},
            {'span': 6.0, 'height': 4.0, 'brace': 1, 'sec_col': 3, 
             'sec_beam': 3, 'sec_brace': 2, 'udl_w': -3000, 'wind_P': 6000},
        ]
        
        X = params_to_features(params_list)
        
        assert X.shape == (2, 8)
        assert X[0, 0] == 5.0  # span
        assert X[1, 2] == 1    # brace


class TestSurrogateModel:
    """Tests for the surrogate model training pipeline."""
    
    def test_model_trains_and_predicts(self):
        """Test that a model can be trained and makes predictions."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 50
        X = np.random.randn(n_samples, 8)
        y = np.random.randn(n_samples)
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Check predictions are finite
        assert np.all(np.isfinite(y_pred))
        assert y_pred.shape == (n_samples,)
    
    def test_model_feature_importances(self):
        """Test that feature importances can be extracted."""
        from sklearn.ensemble import RandomForestRegressor
        
        np.random.seed(42)
        X = np.random.randn(50, 8)
        y = np.random.randn(50)
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        importances = model.feature_importances_
        
        assert len(importances) == 8
        assert np.isclose(importances.sum(), 1.0)
        assert np.all(importances >= 0)


class TestIntegration:
    """Integration tests using real explore module."""
    
    def test_evaluate_variant_returns_dict(self):
        """Test that evaluate_variant returns expected dict structure."""
        from mini_branch.explore import PortalParams, evaluate_variant
        from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
        
        params = PortalParams(
            span=6.0,
            height=3.0,
            brace=1,
            sec_col=3,
            sec_beam=4,
            sec_brace=2,
            udl_w=-2000.0,
            wind_P=5000.0,
            shipping_limit=12.0,
        )
        
        result = evaluate_variant(params, DEFAULT_MATERIAL, TIMBER_SECTIONS)
        
        # Check all expected keys
        assert 'span' in result
        assert 'drift' in result
        assert 'volume' in result
        assert 'ok' in result
        assert result['ok'] == True
        assert np.isfinite(result['drift'])
    
    def test_sample_params_generates_valid_variants(self):
        """Test that sample_params generates valid PortalParams."""
        from mini_branch.explore import sample_params
        from mini_branch.catalog import TIMBER_SECTIONS
        
        rng = np.random.default_rng(seed=42)
        variants = sample_params(rng, TIMBER_SECTIONS, n=10)
        
        assert len(variants) == 10
        
        for v in variants:
            assert v.span > 0
            assert v.height > 0
            assert v.brace in [0, 1]
            assert 0 <= v.sec_col < len(TIMBER_SECTIONS)


class TestEndToEnd:
    """End-to-end test of the ML pipeline (minimal version)."""
    
    def test_mini_ml_pipeline(self):
        """
        Run a minimal version of the ML pipeline:
        1. Generate a few designs
        2. Train a tiny model
        3. Make predictions
        """
        from mini_branch.explore import sample_params, evaluate_variant
        from mini_branch.catalog import TIMBER_SECTIONS, DEFAULT_MATERIAL
        from sklearn.ensemble import RandomForestRegressor
        
        # Generate small dataset
        rng = np.random.default_rng(seed=42)
        variants = sample_params(rng, TIMBER_SECTIONS, n=30)
        
        # Evaluate
        results = []
        for v in variants:
            result = evaluate_variant(v, DEFAULT_MATERIAL, TIMBER_SECTIONS)
            if result['ok']:
                results.append(result)
        
        assert len(results) >= 20, "Should have at least 20 successful designs"
        
        # Build feature matrix
        feature_cols = ['span', 'height', 'brace', 'sec_col', 'sec_beam', 
                        'sec_brace', 'udl_w', 'wind_P']
        X = np.array([[r[c] for c in feature_cols] for r in results])
        y = np.array([r['drift'] for r in results])
        
        # Train tiny model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Predict
        y_pred = model.predict(X)
        
        # Basic sanity checks
        assert np.all(np.isfinite(y_pred))
        assert y_pred.shape == y.shape
        
        # Model should do better than predicting the mean
        mean_baseline_error = np.mean(np.abs(y - y.mean()))
        model_error = np.mean(np.abs(y - y_pred))
        assert model_error < mean_baseline_error, "Model should beat mean baseline"



