"""Identificador interativo de espécies Iris com árvore de decisão."""

import math

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROTULOS_MEDIDAS = (
    "Comprimento da sépala",
    "Largura da sépala",
    "Comprimento da pétala",
    "Largura da pétala",
)


def treinar_modelo():
    """Treina o modelo com 80% do Iris; mantém 20% fora do treinamento."""
    iris = load_iris()

    X_desenvolvimento, _, y_desenvolvimento, _ = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    # Profundidade escolhida no laboratório por validação cruzada 5-fold.
    modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo.fit(X_desenvolvimento, y_desenvolvimento)

    limites_minimos = X_desenvolvimento.min(axis=0)
    limites_maximos = X_desenvolvimento.max(axis=0)

    return modelo, iris.target_names, limites_minimos, limites_maximos


def ler_medida(rotulo):
    """Lê um número positivo, aceitando ponto ou vírgula decimal."""
    while True:
        entrada = input(f"{rotulo} (cm): ").strip().replace(",", ".")

        try:
            valor = float(entrada)
        except ValueError:
            print("Valor inválido. Digite um número, por exemplo: 5.2")
            continue

        if not math.isfinite(valor) or valor <= 0:
            print("A medida precisa ser um número positivo e finito.")
            continue

        return valor


def identificar_flor(modelo, nomes_especies, limites_minimos, limites_maximos):
    """Lê quatro medidas e apresenta a espécie e probabilidades estimadas."""
    print("\n=== IDENTIFICADOR DE FLORES ===")
    print("Informe as quatro medidas da flor em centímetros.\n")

    medidas = [ler_medida(rotulo) for rotulo in ROTULOS_MEDIDAS]
    nova_flor = [medidas]  # Formato 2D: uma amostra com quatro features.

    fora_da_faixa = [
        rotulo
        for rotulo, valor, minimo, maximo in zip(
            ROTULOS_MEDIDAS, medidas, limites_minimos, limites_maximos
        )
        if valor < minimo or valor > maximo
    ]

    if fora_da_faixa:
        print("\nAVISO: medidas fora das faixas observadas no treinamento:")
        for rotulo in fora_da_faixa:
            print(f"- {rotulo}")
        print("A previsão pode ser pouco confiável.\n")

    indice = int(modelo.predict(nova_flor)[0])
    probabilidades = modelo.predict_proba(nova_flor)[0]

    print("\n=== RESULTADO ===")
    print(f"Espécie prevista: {nomes_especies[indice]}")
    print("\nProbabilidades estimadas pelo modelo:")

    for nome, probabilidade in zip(nomes_especies, probabilidades):
        print(f"{nome}: {probabilidade:.2%}")

    print("\nNota: essas estimativas não são garantias de acerto.")


def main():
    modelo, nomes_especies, limites_minimos, limites_maximos = treinar_modelo()
    identificar_flor(modelo, nomes_especies, limites_minimos, limites_maximos)


if __name__ == "__main__":
    main()
