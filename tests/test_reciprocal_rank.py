import pytest
from src.evaluation.metrics import get_reciprocal_rank

# For now we want tiebreakers to return the mean rank of all the tied elements (so the score is 1/mean rank). 
@pytest.mark.parametrize("pos, neg, expected", [
    # Test case 1: Simple case with one positive at the top
    ([1.0], [0.5, 0.3], 1.0),
    
    # Test case 2: Simple case with one positive not at the top
    ([0.7], [0.9, 0.5], 0.5),
    
    # Test case 3: Case with a tie at the top
    ([0.2], [0.2, 0.0, 0.0], 0.666666),
    
    # Test case 4: Multiple positives
    ([0.9, 0.7, 0.5], [0.8, 0.6], 1.0),
    
    # Test case 5: All negatives
    ([], [0.5, 0.3, 0.1], 0),
    
    # Test case 6: All positives
    ([0.9, 0.7, 0.5], [], 1.0),
    
    # Test case 7: All similarities are the same
    ([0.5, 0.5], [0.5, 0.5, 0.5], 1 / 3),
    
    # Test case 8: Positive at the bottom
    ([0.1], [0.9, 0.8, 0.7, 0.6], 0.2),
    
    # Test case 9
    ([0.9, 0.7, 0.7, 0.5], [0.8, 0.7, 0.6, 0.5, 0.5], 1.0)
])
def test_reciprocal_rank(pos, neg, expected):
    result = get_reciprocal_rank(pos, neg)
    if expected is None:
        assert result is None
    else:
        assert pytest.approx(result, abs=1e-6) == expected