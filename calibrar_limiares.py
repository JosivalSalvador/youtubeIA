"""
calibrar_limiares.py
=====================
Recalcula e salva os limiares fixos usados por score_viral, label_viral
e estrutura_blocos — para que essas features deixem de depender do lote
de vídeos sendo processado no momento e passem a usar uma referência
estável, "congelada" a partir da base de produção atual.

Os limiares são calculados POR NICHO: cada nicho tem sua própria escala
de engajamento (ex: "satisfying" e "tech_review" não jogam no mesmo
campo de views), então cada vídeo é comparado só contra o histórico do
seu próprio nicho — nunca contra a base inteira misturada.

Uso:
    Rodar manualmente sempre que quiser recalibrar (ex: depois que a base
    de produção cresceu ou mudou), ou no fim da execução diária do
    orquestrador.

Fonte dos dados:
    Se 'dataset_youtube_processado_modulo2.csv' já existir fisicamente na
    pasta (ex: orquestrador acabou de rodar e ainda não limpou), este
    script reaproveita o arquivo sem baixar de novo. Caso contrário, baixa
    a versão mais recente do Supabase via baixar_csv_mestre() — mesmo
    padrão usado pelo orquestrador — e apaga o CSV físico ao final
    (só o que ele mesmo baixou; nunca mexe em um CSV que já estava lá).

Saída   : limiares_viralidade.json (mesma pasta)
          Formato: { "<nicho>": { "score_viral": {...}, "label_viral": {...},
                                   "estrutura_blocos": {...}, "metadados": {...} },
                     "<outro_nicho>": { ... }, ... }

Os limiares calculados aqui são os MESMOS números que score_viral,
label_viral e estrutura_blocos hoje recalculam a cada execução a partir
do df local — a diferença é que agora eles são calculados 1x por nicho,
sobre a base de produção inteira, e salvos para uso fixo no
processamento de cada vídeo individual.
"""

import json
import os

import pandas as pd

from conexoes_externas import baixar_csv_mestre, NOME_ARQUIVO_MESTRE

# ─────────────────────────────────────────────────────────────
# CAMINHOS
# ─────────────────────────────────────────────────────────────

NOME_ARQUIVO_LIMIARES = "limiares_viralidade.json"

CAMINHO_MESTRE = os.path.join(os.path.dirname(__file__), NOME_ARQUIVO_MESTRE)
CAMINHO_LIMIARES = os.path.join(os.path.dirname(__file__), NOME_ARQUIVO_LIMIARES)


# ─────────────────────────────────────────────────────────────
# CÁLCULO DOS LIMIARES
# ─────────────────────────────────────────────────────────────

def calcular_limiares_de_um_nicho(df: pd.DataFrame) -> dict:
    """
    Calcula, sobre o DataFrame recebido (já filtrado para UM ÚNICO nicho),
    todos os números fixos que as 3 features dependentes de lote
    (score_viral, label_viral, estrutura_blocos) vão passar a consultar
    em vez de recalcular.

    IMPORTANTE: para os quartis de score_viral (usados no label_viral),
    o score_viral aqui é recalculado uma única vez com min/max do
    próprio df recebido — é o único lugar do pipeline onde isso
    acontece, porque é exatamente o ato de calibração.
    """
    # ── 1. score_viral — min/max das 3 métricas que compõem o score ────────
    colunas_score = ["velocidade_views", "taxa_conversao", "taxa_discussao"]
    faltando = [c for c in colunas_score if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes para calibrar score_viral: {faltando}")

    limiares_score_viral = {
        col: {"min": float(df[col].min()), "max": float(df[col].max())}
        for col in colunas_score
    }

    # ── 2. label_viral — cortes de quartil sobre o score_viral calibrado ───
    peso_views, peso_conversao, peso_discussao = 0.60, 0.30, 0.10

    def minmax(serie, minimo, maximo):
        if maximo == minimo:
            return pd.Series(0.0, index=serie.index)
        return (serie - minimo) / (maximo - minimo)

    views_norm = minmax(df["velocidade_views"], limiares_score_viral["velocidade_views"]["min"], limiares_score_viral["velocidade_views"]["max"])
    conversao_norm = minmax(df["taxa_conversao"], limiares_score_viral["taxa_conversao"]["min"], limiares_score_viral["taxa_conversao"]["max"])
    discussao_norm = minmax(df["taxa_discussao"], limiares_score_viral["taxa_discussao"]["min"], limiares_score_viral["taxa_discussao"]["max"])

    score_viral_calibracao = (
        (views_norm * peso_views)
        + (conversao_norm * peso_conversao)
        + (discussao_norm * peso_discussao)
    )

    cortes_quartil = score_viral_calibracao.quantile([0.25, 0.50, 0.75]).tolist()

    limiares_label_viral = {
        "cortes_score_viral": {
            "q25": float(cortes_quartil[0]),
            "q50": float(cortes_quartil[1]),
            "q75": float(cortes_quartil[2]),
        },
        "labels_em_ordem": ["frio", "aquecido", "viral", "super_viral"],
    }

    # ── 3. estrutura_blocos — quartis de densidade_roteiro e ritmo ─────────
    colunas_estrutura = ["densidade_roteiro", "ritmo_palavras_seg"]
    faltando = [c for c in colunas_estrutura if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes para calibrar estrutura_blocos: {faltando}")

    limiares_estrutura_blocos = {
        "densidade_roteiro": {
            "q1": float(df["densidade_roteiro"].quantile(0.25)),
            "q3": float(df["densidade_roteiro"].quantile(0.75)),
        },
        "ritmo_palavras_seg": {
            "q1": float(df["ritmo_palavras_seg"].quantile(0.25)),
            "q3": float(df["ritmo_palavras_seg"].quantile(0.75)),
        },
    }

    return {
        "score_viral": limiares_score_viral,
        "label_viral": limiares_label_viral,
        "estrutura_blocos": limiares_estrutura_blocos,
        "metadados": {
            "n_videos_calibracao": int(len(df)),
            "calibrado_em": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    }


def calcular_limiares_todos_nichos(df: pd.DataFrame) -> dict:
    """
    Agrupa o DataFrame por nicho e calcula os limiares de cada um
    separadamente — cada nicho é comparado só contra o próprio
    histórico, nunca contra a base inteira misturada.

    Nichos sem dados suficientes (ex: < 4 vídeos, insuficiente para
    formar quartis com sentido) são pulados com aviso, em vez de
    quebrar a calibração inteira.
    """
    if "nicho" not in df.columns:
        raise ValueError("Coluna 'nicho' ausente — não é possível calibrar por nicho.")

    resultado = {}

    for nicho, df_nicho in df.groupby("nicho"):
        if len(df_nicho) < 4:
            print(f"[Calibração]   ⚠️  Nicho '{nicho}' pulado: apenas {len(df_nicho)} vídeo(s), insuficiente para calibrar.")
            continue

        print(f"[Calibração]   Calibrando nicho '{nicho}' ({len(df_nicho)} vídeos)...")
        resultado[nicho] = calcular_limiares_de_um_nicho(df_nicho)

    return resultado


# ─────────────────────────────────────────────────────────────
# LEITURA DE LIMIARES (usada por processador_features.py e 00_ajuste.py)
# ─────────────────────────────────────────────────────────────

def carregar_limiares() -> dict:
    """
    Lê o JSON de limiares já calibrados — um bloco por nicho.
    Use carregar_limiares()[nicho] para pegar os limiares de um nicho
    específico. Lança erro claro se o arquivo ainda não existir, ou se
    o nicho pedido não tiver sido calibrado.
    """
    if not os.path.exists(CAMINHO_LIMIARES):
        raise FileNotFoundError(
            f"'{NOME_ARQUIVO_LIMIARES}' não encontrado em {CAMINHO_LIMIARES}. "
            "Rode 'python calibrar_limiares.py' pelo menos uma vez antes de "
            "processar vídeos."
        )

    with open(CAMINHO_LIMIARES, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # Se o CSV já existe fisicamente (ex: orquestrador rodando/acabou de rodar
    # e ainda não limpou), reaproveita — evita baixar de novo à toa.
    # Caso contrário, baixa do Supabase como o orquestrador faz, e ao final
    # apaga o arquivo físico que ESTE script baixou (não mexe em nada que já
    # estava lá antes de rodar).
    csv_ja_existia = os.path.exists(CAMINHO_MESTRE)

    if csv_ja_existia:
        print(f"[Calibração] CSV local encontrado, reaproveitando: {CAMINHO_MESTRE}")
        df = pd.read_csv(CAMINHO_MESTRE)
    else:
        print("[Calibração] CSV local não encontrado. Baixando do Supabase...")
        df = baixar_csv_mestre()

    if df is None or df.empty:
        print("[Calibração] ERRO: base de produção está vazia ou não pôde ser carregada. Abortando.")
        return

    print(f"[Calibração] {len(df)} vídeos carregados. Calculando limiares por nicho...")

    limiares_por_nicho = calcular_limiares_todos_nichos(df)

    with open(CAMINHO_LIMIARES, "w", encoding="utf-8") as f:
        json.dump(limiares_por_nicho, f, ensure_ascii=False, indent=2)

    print(f"[Calibração] ✓ Limiares salvos em: {CAMINHO_LIMIARES}")
    print(f"[Calibração]   Nichos calibrados: {list(limiares_por_nicho.keys())}")

    # A vassoura — só apaga o CSV físico se foi ESTE script quem baixou.
    # Se o orquestrador deixou o arquivo ali (ex: chamando calibrar logo após
    # a Fase 3, antes de rodar sua própria limpeza), não mexemos nele.
    if not csv_ja_existia and os.path.exists(CAMINHO_MESTRE):
        os.remove(CAMINHO_MESTRE)
        print(f"[Calibração] 🧹 Limpeza: '{NOME_ARQUIVO_MESTRE}' (baixado por este script) apagado.")


if __name__ == "__main__":
    main()