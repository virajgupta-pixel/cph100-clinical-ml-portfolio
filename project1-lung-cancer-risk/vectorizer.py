import numpy as np
from collections import Counter

def _is_missing(x):
    if x is None:
        return True
    s = str(x).strip()
    if s == "":
        return True
    ls = s.lower()
    if ls in ("na", "nan", "missing", "unknown", "none", "n/a"):
        return True
    return False

def _to_float_or_nan(x):
    try:
        if _is_missing(x):
            return np.nan
        return float(str(x).strip())
    except Exception:
        return np.nan

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False
        self.num_bins = num_bins

    def get_numerical_vectorizer(self, values, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        parsed = np.array([_to_float_or_nan(v) for v in values], dtype=float)

        if np.all(np.isnan(parsed)):
            mean = 0.0
            std = 1.0
        else:
            mean = float(np.nanmean(parsed))
            std = float(np.nanstd(parsed))
            if std == 0.0 or np.isnan(std):
                std = 1.0

        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            xv = _to_float_or_nan(x)
            missing = 1.0 if np.isnan(xv) else 0.0
            if np.isnan(xv):
                xv = mean
            val = (xv - mean) / std
            return np.array([val, missing], dtype=float)

        return vectorizer, mean, std

    def get_categorical_vectorizer(self, values):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        cats_raw = []
        for v in values:
            if _is_missing(v):
                continue
            cats_raw.append(str(v).strip())
        
        unique_cats = sorted(set(cats_raw))
        missing_token = "__MISSING__"
        categories = unique_cats + [missing_token]
        mapping = {c: i for i, c in enumerate(categories)}
        missing_idx = mapping[missing_token]

        def vectorizer(x):
            if _is_missing(x):
                idx = missing_idx
            else:
                key = str(x).strip()
                idx = mapping.get(key, missing_idx)
            vec = np.zeros(len(categories), dtype=float)
            vec[idx] = 1.0
            return vec

        return vectorizer, categories
    
    def get_ordinal_vectorizer(self, values):
        non_missing = [v for v in values if not _is_missing(v)]
        numeric_converted = []
        numeric_ok = True
        for v in non_missing:
            try:
                numeric_converted.append(float(str(v).strip()))
            except Exception:
                numeric_ok = False
                break
        
        if numeric_ok and len(numeric_converted) > 0:
            unique_vals = sorted(set(numeric_converted))
            mapping = {str(val): i for i, val in enumerate(unique_vals)}
            counts = Counter(numeric_converted)
            mode_val = float(counts.most_common(1)[0][0])
            impute_index = mapping[str(mode_val)]
            def vectorizer(x):
                if _is_missing(x):
                    return np.array([float(impute_index), 1.0], dtype=float)
                try:
                    xv = float(str(x).strip())
                except Exception:
                    xv = mode_val
                idx = mapping.get(str(xv), impute_index)
                return np.array([float(idx), 0.0], dtype=float)
            meta = {"mapping": mapping, "mode": mode_val, "dim": 2}
            return vectorizer, meta
        else:
            unique_vals = sorted(set([str(v).strip() for v in non_missing]))
            mapping = {v: i for i, v in enumerate(unique_vals)}
            if len(non_missing) > 0:
                counts = Counter([str(v).strip() for v in non_missing])
                mode_val = counts.most_common(1)[0][0]
                impute_index = mapping[mode_val]
            else:
                mapping = {}
                mode_val = ""
                impute_index = 0

            def vectorizer(x):
                if _is_missing(x):
                    return np.array([float(impute_index), 1.0], dtype=float)
                key = str(x).strip()
                idx = mapping.get(key, impute_index)
                return np.array([float(idx), 0.0], dtype=float)

            meta = {"mapping": mapping, "mode": mode_val, "dim": 2}
            return vectorizer, meta

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        
        numerical = self.feature_config.get("numerical", [])
        for feat in numerical:
            values = [r.get(feat, None) for r in X]
            vec_fn, mean, std = self.get_numerical_vectorizer(values)
            self.feature_transforms[feat] = ("numerical", vec_fn, {"mean": mean, "std": std})
        
        categorical = self.feature_config.get("categorical", []) or []
        for feat in categorical:
            values = [r.get(feat, None) for r in X]
            vec_fn, categories = self.get_categorical_vectorizer(values)
            meta = {"categories": categories, "dim": len(categories)}
            self.feature_transforms[feat] = ("categorical", vec_fn, meta)
        
        ordinal = self.feature_config.get("ordinal", []) or []
        for feat in ordinal:
            values = [r.get(feat, None) for r in X]
            vec_fn, meta = self.get_ordinal_vectorizer(values)
            self.feature_transforms[feat] = ("ordinal", vec_fn, meta)

        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        rows = []
        numerical = self.feature_config.get("numerical", []) or []
        categorical = self.feature_config.get("categorical", []) or []
        ordinal = self.feature_config.get("ordinal", []) or []

        for r in X:
            feats = []
            for feat in numerical:
                typ, vec_fn, meta = self.feature_transforms[feat]
                out = vec_fn(r.get(feat, None))
                out_arr = np.asarray(out, dtype=float).ravel().tolist()
                feats.extend(out_arr)

            for feat in categorical:
                typ, vec_fn, meta = self.feature_transforms[feat]
                out = vec_fn(r.get(feat, None))
                out_arr = np.asarray(out, dtype=float).ravel().tolist()
                feats.extend(out_arr)

            for feat in ordinal:
                typ, vec_fn, meta = self.feature_transforms[feat]
                out = vec_fn(r.get(feat, None))
                out_arr = np.asarray(out, dtype=float).ravel().tolist()
                feats.extend(out_arr)

            rows.append(feats)

        return np.array(rows, dtype=float)