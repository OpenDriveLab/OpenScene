import unittest

from nuplan.planning.simulation.callback.test.skeleton_test_serialization_callback import (
    SkeletonTestSerializationCallback,
)


class TestSerializationCallbackJson(SkeletonTestSerializationCallback):
    """Tests that SerializationCallback works correctly for JSON format."""

    def setUp(self) -> None:
        """Will be called before every test"""
        self._serialization_type = "json"

        self._setUp()

    def test_serialization_callback(self) -> None:
        """Tests that we can correctly serialize data to JSON format."""
        self._dump_test_scenario()


if __name__ == '__main__':
    unittest.main()
