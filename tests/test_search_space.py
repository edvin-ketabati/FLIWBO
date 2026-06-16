import numpy as np
import pytest

from fliwbo_core import Continuous, Discrete, SearchSpace
from fliwbo_core.optimizer import FLIWBOOptimizer


def test_discrete_coordinates_normalize_to_expected_model_domain():
    space = SearchSpace([Discrete(4), Discrete(3)])

    normalized = space.normalize_matrix(np.array([[0, 1], [3, 2], [5, -1]]))

    np.testing.assert_allclose(
        normalized,
        np.array([
            [0.02, 0.50],
            [0.98, 0.98],
            [0.98, 0.02],
        ]),
    )


def test_mixed_space_projects_only_needed_coordinates():
    space = SearchSpace([Discrete(4), Continuous(-2.0, 2.0)])

    projected = space.project_vector(np.array([3.7, 4.0]))

    assert projected[0] == 3
    assert projected[1] == 2.0


def test_jsonable_vector_preserves_python_number_types():
    space = SearchSpace([Discrete(4), Continuous(-2.0, 2.0)])

    values = space.vector_to_jsonable(np.array([1.2, 0.25]))

    assert values == [1, 0.25]
    assert isinstance(values[0], int)
    assert isinstance(values[1], float)


def test_manifest_round_trip_uses_typed_variables_only():
    original = SearchSpace([Discrete(4), Continuous(-2.0, 2.0)])

    restored = SearchSpace.from_jsonable(original.to_jsonable())

    assert restored.dimension == 2
    assert isinstance(restored.variables[0], Discrete)
    assert isinstance(restored.variables[1], Continuous)


def test_optimizer_requires_explicit_search_space():
    with pytest.raises(TypeError, match="SearchSpace"):
        FLIWBOOptimizer([4, 3])
