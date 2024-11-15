import unittest
from WarehouseEnv import WarehouseEnv, Robot, Package, ChargeStation, manhattan_distance
from submission import smart_heuristic


class TestSmartHeuristic(unittest.TestCase):
    def setUp(self):
        self.env = WarehouseEnv()
        self.env.width = 5
        self.env.height = 5
        self.env.robots = [Robot((0, 0), 20, 0), Robot((4, 4), 10, 5)]
        self.env.packages = [
            Package((1, 1), (3, 3)),
            Package((2, 2), (4, 4))
        ]
        for pkg in self.env.packages:
            pkg.on_board = True
        self.env.charge_stations = [ChargeStation((0, 1)), ChargeStation((4, 3))]

    def test_smart_heuristic(self):
        heuristic_value_0 = smart_heuristic(self.env, 0)
        heuristic_value_1 = smart_heuristic(self.env, 1)

        # Print heuristic values
        print("Heuristic value for robot 0:", heuristic_value_0)
        print("Heuristic value for robot 1:", heuristic_value_1)

        # Basic assertions to check if heuristic values are within expected range
        self.assertGreaterEqual(heuristic_value_0, 0)
        self.assertLessEqual(heuristic_value_0, 10000)
        self.assertGreaterEqual(heuristic_value_1, 0)
        self.assertLessEqual(heuristic_value_1, 10000)

if __name__ == "__main__":
    unittest.main()
