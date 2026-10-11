"""Testes básicos da aplicação, utilizando unittest da biblioteca padrão."""

import unittest

from main import treinar_modelo


class TestIdentificadorIris(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.modelo, cls.nomes, cls.minimos, cls.maximos = treinar_modelo()

    def test_tres_especies(self):
        self.assertEqual(list(self.nomes), ["setosa", "versicolor", "virginica"])

    def test_flor_setosa(self):
        previsao = int(self.modelo.predict([[5.2, 3.6, 1.5, 0.3]])[0])
        self.assertEqual(self.nomes[previsao], "setosa")

    def test_probabilidades_somam_um(self):
        probabilidades = self.modelo.predict_proba([[5.2, 3.6, 1.5, 0.3]])[0]
        self.assertAlmostEqual(float(sum(probabilidades)), 1.0)

    def test_quatro_caracteristicas(self):
        self.assertEqual(len(self.minimos), 4)
        self.assertEqual(len(self.maximos), 4)


if __name__ == "__main__":
    unittest.main()
