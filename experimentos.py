"""Reproduz a comparação de profundidades e gera imagens para o README."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Geração de gráficos sem abrir uma janela.
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


PROFUNDIDADES = [1, 2, 3, 4, None]
DIRETORIO_IMAGENS = Path(__file__).resolve().parent / "assets"


def avaliar_profundidades():
    iris = load_iris()

    # 20% ficam reservados; não são usados para selecionar hiperparâmetros.
    X_dev, _, y_dev, _ = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    # Separação simples para comparar com a validação cruzada.
    X_treino, X_val, y_treino, y_val = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )

    divisoes = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []

    for profundidade in PROFUNDIDADES:
        modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
        modelo.fit(X_treino, y_treino)

        acuracia_treino = accuracy_score(y_treino, modelo.predict(X_treino))
        acuracia_validacao = accuracy_score(y_val, modelo.predict(X_val))

        acuracias_cv = cross_val_score(
            DecisionTreeClassifier(max_depth=profundidade, random_state=42),
            X_dev,
            y_dev,
            cv=divisoes,
            scoring="accuracy",
        )

        resultados.append(
            {
                "profundidade": "None" if profundidade is None else str(profundidade),
                "profundidade_real": modelo.get_depth(),
                "treino": acuracia_treino,
                "validacao": acuracia_validacao,
                "cv_media": acuracias_cv.mean(),
                "cv_rodadas": acuracias_cv.tolist(),
            }
        )

    return iris, X_dev, y_dev, resultados


def gerar_arquivos(iris, X_dev, y_dev, resultados):
    DIRETORIO_IMAGENS.mkdir(exist_ok=True)

    # Tabela tabular dos experimentos, para reproduzir os números do README.
    with (DIRETORIO_IMAGENS / "resultados.csv").open(
        "w", newline="", encoding="utf-8"
    ) as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(
            ["max_depth", "profundidade_real", "treino", "validacao", "cv_5fold"]
        )
        for r in resultados:
            escritor.writerow(
                [
                    r["profundidade"],
                    r["profundidade_real"],
                    f'{r["treino"]:.6f}',
                    f'{r["validacao"]:.6f}',
                    f'{r["cv_media"]:.6f}',
                ]
            )

    # Gráfico das avaliações com um conjunto de validação e com 5 folds.
    labels = [r["profundidade"] for r in resultados]
    valores_validacao = [100 * r["validacao"] for r in resultados]
    valores_cv = [100 * r["cv_media"] for r in resultados]
    posicoes = np.arange(len(labels))
    largura = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    barras_val = ax.bar(posicoes - largura / 2, valores_validacao, largura, label="Validação simples")
    barras_cv = ax.bar(posicoes + largura / 2, valores_cv, largura, label="Validação cruzada (5 folds)")
    ax.set_xticks(posicoes, labels)
    ax.set_xlabel("Profundidade máxima (max_depth)")
    ax.set_ylabel("Acurácia (%)")
    ax.set_ylim(55, 105)
    ax.set_title("Iris · Comparação de profundidades da árvore")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for barras in [barras_val, barras_cv]:
        for barra in barras:
            ax.annotate(
                f"{barra.get_height():.2f}%".replace(".", ","),
                (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 4),
                textcoords="offset points",
            )
    fig.tight_layout()
    fig.savefig(DIRETORIO_IMAGENS / "comparacao-modelos.png", dpi=180)
    plt.close(fig)

    # Árvore final escolhida pela maior média de validação cruzada.
    melhor = max(resultados, key=lambda item: item["cv_media"])
    profundidade = (
        None if melhor["profundidade"] == "None" else int(melhor["profundidade"])
    )
    modelo_final = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
    modelo_final.fit(X_dev, y_dev)

    fig, ax = plt.subplots(figsize=(18, 8.5))
    plot_tree(
        modelo_final,
        feature_names=list(iris.feature_names),
        class_names=list(iris.target_names),
        filled=True,
        rounded=True,
        impurity=False,
        ax=ax,
        fontsize=10,
    )
    ax.set_title("Árvore de decisão treinada com 120 flores (max_depth=3)", fontsize=15)
    fig.tight_layout()
    fig.savefig(DIRETORIO_IMAGENS / "arvore-decisao.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return melhor


def main():
    iris, X_dev, y_dev, resultados = avaliar_profundidades()

    print("\n--- COMPARAÇÃO DOS MODELOS ---")
    print(f'{"Depth":>7} | {"Treino":>9} | {"Validação":>9} | {"CV 5-fold":>9}')
    print("-" * 47)
    for r in resultados:
        print(
            f'{r["profundidade"]:>7} | {r["treino"]:>9.2%} | '
            f'{r["validacao"]:>9.2%} | {r["cv_media"]:>9.2%}'
        )

    melhor = gerar_arquivos(iris, X_dev, y_dev, resultados)
    print(f'\nProfundidade selecionada por CV: {melhor["profundidade"]}')
    print("Gráficos e tabela salvos em assets/.")


if __name__ == "__main__":
    main()
