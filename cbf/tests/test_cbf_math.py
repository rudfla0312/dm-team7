"""Small hand-checkable calculation kept outside the final Colab notebook."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_helpers import nearest_diseases, rank_drugs, make_fitting_data


def test_hand_calculated_cosine_and_scores():
    # A=(1,0), B=(.8,.6), C=(.6,.8), D=(0,1); A·B=.8 and A·C=.6.
    diseases = ['A', 'B', 'C', 'D']
    vectors = np.array([[1,0],[.8,.6],[.6,.8],[0,1.]])
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, 0.0)
    fit = pd.DataFrame([('A','Y'),('B','X'),('C','X'),('C','Y'),('D','Z')],
                       columns=['disease_id','drug_id'])
    index = {d:i for i,d in enumerate(diseases)}
    model = make_fitting_data(fit, index)
    nearest = nearest_diseases('A', similarity, model, diseases, index)
    assert [d for d,_ in nearest[:2]] == ['B','C']
    assert np.isclose(nearest[0][1], .8)
    assert np.isclose(nearest[1][1], .6)
    ranked = rank_drugs('A', nearest, model, {}, 2, 2)
    # Y is an existing indication of A and is masked. X receives .8²+.6²=1.
    assert len(ranked) == 1 and ranked[0][0] == 'X'
    assert np.isclose(ranked[0][1], 1.0)
    assert rank_drugs('A', nearest, model, {'A':{'X'}}, 2, 2) == []


if __name__ == '__main__':
    test_hand_calculated_cosine_and_scores()
    print('Manual cosine / neighbor / p=2 score / conflict masking: PASS')
