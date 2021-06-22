import numpy as np
import pandas as pd

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal
}

def Query(table, columns, operators, vals, return_masks=False, return_crad_and_masks=False):
        assert len(columns) == len(operators) == len(vals)
        bools = None
        for c, o, v in zip(columns, operators, vals):
            if table.name in ['Adult', 'Census']: 
                inds = [False] * table.cardinality
                inds = np.array(inds)
                is_nan = pd.isnull(c.data)
                if np.any(is_nan):
                    inds[~is_nan] = OPS[o](c.data[~is_nan], v)
                else:
                    inds = OPS[o](c.data, v)
            else:
                inds = OPS[o](c.data, v)

            if bools is None:
                bools = inds
            else:
                bools &= inds
        c = bools.sum()
        if return_masks:
            return bools
        elif return_crad_and_masks:
            return c, bools
        return c