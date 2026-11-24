import unittest

from model import predict_price, train_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.metrics, cls.actual, cls.predicted = train_model()

    def test_holdout(self):
        self.assertEqual(self.metrics["train_samples"], 480)
        self.assertEqual(self.metrics["test_samples"], 120)
        self.assertEqual(len(self.predicted), len(self.actual))
        self.assertGreater(self.metrics["r2"], 0.8)

    def test_prediction(self):
        self.assertGreater(predict_price(self.model, 100, 3, "City Center"), 0)

    def test_larger_house_costs_more(self):
        self.assertGreater(predict_price(self.model, 150, 3, "City Center"),
                           predict_price(self.model, 100, 3, "City Center"))

    def test_location_matters(self):
        self.assertGreater(predict_price(self.model, 100, 3, "City Center"),
                           predict_price(self.model, 100, 3, "Outer Suburb"))

    def test_invalid_inputs(self):
        for inputs in [(10, 3, "City Center"), (100, 7, "City Center"),
                       (100, 3, "Unknown")]:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                predict_price(self.model, *inputs)


if __name__ == "__main__":
    unittest.main()
