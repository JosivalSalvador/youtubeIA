"""
verificar_ajuste.py
===================
Auditoria completa do CSV após rodar o 00_ajuste.py.
Verifica estrutura, ordem de colunas, tipos, nulos, domínios
e consistência interna das features — sem depender de valores
esperados fixos (tudo relativo ao próprio dataset).

Uso:
    python verificar_ajuste.py
    python verificar_ajuste.py --csv outro_arquivo.csv

Saída: relatório no console + verificar_ajuste_relatorio.txt no mesmo diretório.
"""

import os
import sys
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────

CSV_DEFAULT = os.path.join(os.path.dirname(__file__), "dataset_youtube_processado_modulo2.csv")

COLUNAS_ORIGINAIS = [
    "video_id", "titulo", "descricao", "tags", "texto_falado",
    "canal_id", "canal_nome", "visualizacoes", "curtidas", "comentarios",
    "data_publicacao", "duracao_segundos", "tem_legenda_nativa",
    "conteudo_licenciado", "feito_para_criancas", "categoria_id",
    "idioma_audio_default", "idioma_texto_default", "topicos_wikipedia",
    "thumb_maxres", "nicho",
]

ORDEM_FEATURES_CRIADAS = [
    "dia_postagem", "hora_postagem", "janela_postagem", "idade_dias",
    "velocidade_views", "taxa_conversao", "taxa_discussao",
    "score_viral", "label_viral",
    "faixa_duracao", "estrutura_blocos", "ritmo_palavras_seg",
    "densidade_roteiro", "tem_repeticao_roteiro", "gancho_primeira_frase",
    "sentimento_roteiro", "tipo_audio_dominante",
    "clickbait_score", "completude_seo", "vibe_emojis",
    "pistas_audio", "pistas_audio_en",
    "palavras_chave", "palavras_chave_en", "vocabulario_falado",
    "texto_falado_limpo", "texto_falado_limpo_en", "titulo_en", "descricao_en",
    "descricao_visual_thumb", "texto_thumbnail",
]

ORDEM_FINAL_ESPERADA = COLUNAS_ORIGINAIS + ORDEM_FEATURES_CRIADAS

# Domínios válidos por coluna categórica
DOMINIOS = {
    "dia_postagem"     : {"monday","tuesday","wednesday","thursday","friday","saturday","sunday"},
    "janela_postagem"  : {"madrugada","manha","tarde","noite"},
    "faixa_duracao"    : {"ultra_curto","short_padrao","short_longo","desconhecido"},
    "estrutura_blocos" : {"impacto_rapido","esquete","esquete_com_hook","narrativa"},
    "label_viral"      : {"frio","aquecido","viral","super_viral"},
    "sentimento_roteiro": {"positivo","negativo","neutro"},
}

# Colunas que NUNCA podem ter nulo após o ajuste
NUNCA_NULAS = [
    "dia_postagem", "hora_postagem", "janela_postagem", "idade_dias",
    "velocidade_views", "taxa_conversao", "taxa_discussao",
    "score_viral", "label_viral", "faixa_duracao", "estrutura_blocos",
    "ritmo_palavras_seg", "densidade_roteiro", "tem_repeticao_roteiro",
    "sentimento_roteiro", "tipo_audio_dominante", "clickbait_score",
    "completude_seo", "vibe_emojis", "pistas_audio", "pistas_audio_en",
    "palavras_chave", "palavras_chave_en", "vocabulario_falado",
    "texto_falado_limpo", "texto_falado_limpo_en",
]

# Colunas numéricas e seus limites lógicos [min, max]
LIMITES_NUMERICOS = {
    "hora_postagem"     : (0, 23),
    "idade_dias"        : (1, 365),
    "velocidade_views"  : (0, None),
    "taxa_conversao"    : (0, 100),
    "taxa_discussao"    : (0, 100),
    "score_viral"       : (0, 1),
    "clickbait_score"   : (0, 1),
    "completude_seo"    : (0, None),
    "ritmo_palavras_seg": (0, None),
    "densidade_roteiro" : (0, None),
}

# ─────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────

class Relatorio:
    """Acumula linhas e no final imprime + salva em arquivo."""

    def __init__(self):
        self.linhas = []
        self.erros  = 0
        self.avisos = 0
        self.oks    = 0

    def ok(self, msg):
        linha = f"  ✅ {msg}"
        self.linhas.append(linha)
        self.oks += 1
        print(linha)

    def erro(self, msg):
        linha = f"  ❌ ERRO   | {msg}"
        self.linhas.append(linha)
        self.erros += 1
        print(linha)

    def aviso(self, msg):
        linha = f"  ⚠️  AVISO  | {msg}"
        self.linhas.append(linha)
        self.avisos += 1
        print(linha)

    def secao(self, titulo):
        sep   = "─" * 60
        linha = f"\n{sep}\n{titulo}\n{sep}"
        self.linhas.append(linha)
        print(linha)

    def salvar(self, caminho):
        with open(caminho, "w", encoding="utf-8") as f:
            f.write("\n".join(self.linhas))
        print(f"\nRelatório salvo em: {caminho}")


# ─────────────────────────────────────────────────────────────
# CHECKS
# ─────────────────────────────────────────────────────────────

def check_colunas_presentes(df: pd.DataFrame, r: Relatorio):
    """Verifica se todas as colunas esperadas existem no CSV."""
    r.secao("1. COLUNAS PRESENTES")
    faltando = [c for c in ORDEM_FINAL_ESPERADA if c not in df.columns]
    extras   = [c for c in df.columns if c not in ORDEM_FINAL_ESPERADA]

    if not faltando:
        r.ok(f"Todas as {len(ORDEM_FINAL_ESPERADA)} colunas esperadas estão presentes.")
    else:
        for c in faltando:
            r.erro(f"Coluna ausente: '{c}'")

    if extras:
        for c in extras:
            r.aviso(f"Coluna inesperada (não mapeada): '{c}'")
    else:
        r.ok("Nenhuma coluna extra inesperada.")


def check_ordem_colunas(df: pd.DataFrame, r: Relatorio):
    """Verifica se a ordem das colunas no CSV é exatamente a esperada."""
    r.secao("2. ORDEM DAS COLUNAS")
    colunas_csv = list(df.columns)
    esperadas   = [c for c in ORDEM_FINAL_ESPERADA if c in df.columns]

    # Compara posição a posição apenas nas colunas que existem
    erros_ordem = []
    pos_csv = {c: i for i, c in enumerate(colunas_csv)}
    for i in range(len(esperadas) - 1):
        a, b = esperadas[i], esperadas[i + 1]
        if a in pos_csv and b in pos_csv:
            if pos_csv[a] > pos_csv[b]:
                erros_ordem.append(f"'{a}' deveria vir antes de '{b}'")

    if not erros_ordem:
        r.ok("Ordem das colunas correta.")
    else:
        for e in erros_ordem:
            r.erro(e)


def check_nulos(df: pd.DataFrame, r: Relatorio):
    """Verifica nulos nas colunas que não podem ter nenhum."""
    r.secao("3. NULOS NAS COLUNAS CRÍTICAS")
    for col in NUNCA_NULAS:
        if col not in df.columns:
            continue
        n_nulos = df[col].isna().sum()
        # Strings vazias também contam como nulo para colunas de texto
        if df[col].dtype == object:
            n_vazios = (df[col].astype(str).str.strip() == "").sum()
        else:
            n_vazios = 0

        if n_nulos == 0 and n_vazios == 0:
            r.ok(f"{col}: sem nulos.")
        else:
            if n_nulos > 0:
                r.erro(f"{col}: {n_nulos} nulos reais (NaN).")
            if n_vazios > 0:
                r.aviso(f"{col}: {n_vazios} strings vazias.")


def check_dominios(df: pd.DataFrame, r: Relatorio):
    """Verifica se valores categóricos estão dentro do domínio esperado."""
    r.secao("4. DOMÍNIOS DAS COLUNAS CATEGÓRICAS")
    for col, dominio in DOMINIOS.items():
        if col not in df.columns:
            continue
        valores = set(df[col].dropna().astype(str).unique())
        invalidos = valores - dominio
        if not invalidos:
            r.ok(f"{col}: todos os valores dentro do domínio {sorted(dominio)}.")
        else:
            r.erro(f"{col}: valores fora do domínio → {sorted(invalidos)}")


def check_limites_numericos(df: pd.DataFrame, r: Relatorio):
    """Verifica se colunas numéricas respeitam os limites lógicos."""
    r.secao("5. LIMITES NUMÉRICOS")
    for col, (minv, maxv) in LIMITES_NUMERICOS.items():
        if col not in df.columns:
            continue
        serie = pd.to_numeric(df[col], errors="coerce")
        if minv is not None and (serie < minv).any():
            n = (serie < minv).sum()
            r.erro(f"{col}: {n} valor(es) abaixo do mínimo esperado ({minv}). Min encontrado: {serie.min():.4f}")
        elif maxv is not None and (serie > maxv).any():
            n = (serie > maxv).sum()
            r.erro(f"{col}: {n} valor(es) acima do máximo esperado ({maxv}). Max encontrado: {serie.max():.4f}")
        else:
            bounds = f"[{minv}, {maxv if maxv is not None else '∞'}]"
            r.ok(f"{col}: valores dentro dos limites {bounds}.")


def check_consistencia_interna(df: pd.DataFrame, r: Relatorio):
    """Verifica consistência lógica entre colunas relacionadas."""
    r.secao("6. CONSISTÊNCIA INTERNA")

    # velocidade_views = visualizacoes / idade_dias (tolerância de float)
    if all(c in df.columns for c in ["velocidade_views", "visualizacoes", "idade_dias"]):
        esperado  = df["visualizacoes"] / df["idade_dias"]
        diff      = (df["velocidade_views"] - esperado).abs()
        errados   = (diff > 0.01).sum()
        if errados == 0:
            r.ok("velocidade_views: consistente com visualizacoes / idade_dias.")
        else:
            r.erro(f"velocidade_views: {errados} linha(s) inconsistentes com visualizacoes / idade_dias.")

    # taxa_conversao = curtidas / visualizacoes * 100
    if all(c in df.columns for c in ["taxa_conversao", "curtidas", "visualizacoes"]):
        mask      = df["visualizacoes"] > 0
        esperado  = (df.loc[mask, "curtidas"] / df.loc[mask, "visualizacoes"]) * 100
        diff      = (df.loc[mask, "taxa_conversao"] - esperado).abs()
        errados   = (diff > 0.01).sum()
        if errados == 0:
            r.ok("taxa_conversao: consistente com curtidas / visualizacoes * 100.")
        else:
            r.erro(f"taxa_conversao: {errados} linha(s) inconsistentes.")

    # taxa_discussao = comentarios / visualizacoes * 100
    if all(c in df.columns for c in ["taxa_discussao", "comentarios", "visualizacoes"]):
        mask      = df["visualizacoes"] > 0
        esperado  = (df.loc[mask, "comentarios"] / df.loc[mask, "visualizacoes"]) * 100
        diff      = (df.loc[mask, "taxa_discussao"] - esperado).abs()
        errados   = (diff > 0.01).sum()
        if errados == 0:
            r.ok("taxa_discussao: consistente com comentarios / visualizacoes * 100.")
        else:
            r.erro(f"taxa_discussao: {errados} linha(s) inconsistentes.")

    # ritmo_palavras_seg: palavras do texto_falado_limpo_en / duracao_segundos
    if all(c in df.columns for c in ["ritmo_palavras_seg", "texto_falado_limpo_en", "duracao_segundos"]):
        contagem  = df["texto_falado_limpo_en"].fillna("").apply(lambda x: len(str(x).split()))
        esperado  = np.where(df["duracao_segundos"] > 0, contagem / df["duracao_segundos"], 0.0)
        diff      = (df["ritmo_palavras_seg"] - esperado).abs()
        errados   = (diff > 0.01).sum()
        if errados == 0:
            r.ok("ritmo_palavras_seg: consistente com palavras / duracao_segundos.")
        else:
            r.erro(f"ritmo_palavras_seg: {errados} linha(s) inconsistentes.")

    # densidade_roteiro: contagem de palavras de texto_falado_limpo_en
    if all(c in df.columns for c in ["densidade_roteiro", "texto_falado_limpo_en"]):
        esperado = df["texto_falado_limpo_en"].fillna("").apply(lambda x: len(str(x).split()))
        errados  = (df["densidade_roteiro"] != esperado).sum()
        if errados == 0:
            r.ok("densidade_roteiro: consistente com len(texto_falado_limpo_en.split()).")
        else:
            r.erro(f"densidade_roteiro: {errados} linha(s) inconsistentes.")

    # gancho_primeira_frase: primeiras 15 palavras de texto_falado_limpo_en
    if all(c in df.columns for c in ["gancho_primeira_frase", "texto_falado_limpo_en"]):
        def gancho(t):
            if pd.isna(t) or not str(t).strip():
                return ""
            return " ".join(str(t).split()[:15])
        esperado = df["texto_falado_limpo_en"].apply(gancho)
        errados  = (df["gancho_primeira_frase"].fillna("") != esperado).sum()
        if errados == 0:
            r.ok("gancho_primeira_frase: consistente com primeiras 15 palavras.")
        else:
            r.erro(f"gancho_primeira_frase: {errados} linha(s) inconsistentes.")

    # faixa_duracao: limites técnicos
    if all(c in df.columns for c in ["faixa_duracao", "duracao_segundos"]):
        cond_ultra  = (df["duracao_segundos"] <= 30) & (df["faixa_duracao"] != "ultra_curto")
        cond_padrao = ((df["duracao_segundos"] > 30) & (df["duracao_segundos"] <= 60)) & (df["faixa_duracao"] != "short_padrao")
        cond_longo  = (df["duracao_segundos"] > 60) & (df["faixa_duracao"] != "short_longo")
        errados     = cond_ultra.sum() + cond_padrao.sum() + cond_longo.sum()
        if errados == 0:
            r.ok("faixa_duracao: limites corretamente aplicados.")
        else:
            r.erro(f"faixa_duracao: {errados} linha(s) com faixa incorreta.")

    # score_viral: deve estar entre 0 e 1 e label_viral deve ser quartil dele
    if all(c in df.columns for c in ["score_viral", "label_viral"]):
        score = pd.to_numeric(df["score_viral"], errors="coerce")
        fora  = ((score < 0) | (score > 1)).sum()
        if fora == 0:
            r.ok("score_viral: todos os valores entre 0 e 1.")
        else:
            r.erro(f"score_viral: {fora} valor(es) fora do intervalo [0,1].")

        # label deve crescer monotonamente com o score
        ordem_label = {"frio": 0, "aquecido": 1, "viral": 2, "super_viral": 3}
        df_tmp = df[["score_viral", "label_viral"]].copy()
        df_tmp["score_viral"] = pd.to_numeric(df_tmp["score_viral"], errors="coerce")
        df_tmp["label_num"]   = df_tmp["label_viral"].map(ordem_label)
        # mediana de score por label deve ser crescente
        medianas = df_tmp.groupby("label_viral", observed=True)["score_viral"].median()
        medianas_ord = medianas.reindex(["frio","aquecido","viral","super_viral"]).dropna()
        if list(medianas_ord) == sorted(medianas_ord):
            r.ok("label_viral: medianas de score crescem corretamente frio→super_viral.")
        else:
            r.erro(f"label_viral: medianas de score fora de ordem → {medianas_ord.to_dict()}")

    # idade_dias: deve ser >= 1 e compatível com data_publicacao
    if all(c in df.columns for c in ["idade_dias", "data_publicacao"]):
        datas     = pd.to_datetime(df["data_publicacao"], utc=True, errors="coerce")
        hoje      = datetime.now(timezone.utc)
        idade_esp = np.clip((hoje - datas).dt.days, 1, 365)
        # Tolerância de 1 dia (pode rodar em dia diferente do ajuste)
        diff      = (df["idade_dias"] - idade_esp).abs()
        errados   = (diff > 1).sum()
        if errados == 0:
            r.ok("idade_dias: consistente com data_publicacao (tolerância ±1 dia).")
        else:
            r.aviso(f"idade_dias: {errados} linha(s) com diferença > 1 dia vs data_publicacao. Normal se o ajuste foi rodado em data diferente.")

    # hora_postagem deve bater com data_publicacao
    if all(c in df.columns for c in ["hora_postagem", "data_publicacao"]):
        datas   = pd.to_datetime(df["data_publicacao"], utc=True, errors="coerce")
        hora_esp = datas.dt.hour
        errados  = (df["hora_postagem"] != hora_esp).sum()
        if errados == 0:
            r.ok("hora_postagem: consistente com data_publicacao.")
        else:
            r.erro(f"hora_postagem: {errados} linha(s) inconsistentes com data_publicacao.")

    # dia_postagem deve bater com data_publicacao
    if all(c in df.columns for c in ["dia_postagem", "data_publicacao"]):
        datas   = pd.to_datetime(df["data_publicacao"], utc=True, errors="coerce")
        dia_esp = datas.dt.day_name().str.lower()
        errados = (df["dia_postagem"] != dia_esp).sum()
        if errados == 0:
            r.ok("dia_postagem: consistente com data_publicacao.")
        else:
            r.erro(f"dia_postagem: {errados} linha(s) inconsistentes com data_publicacao.")

    # texto_falado_limpo: não pode conter tags [..] ou (...)
    if "texto_falado_limpo" in df.columns:
        tem_tag = df["texto_falado_limpo"].astype(str).str.contains(r"\[.*?\]|\(.*?\)|♪", regex=True, na=False)
        n = tem_tag.sum()
        if n == 0:
            r.ok("texto_falado_limpo: nenhuma tag ASR residual encontrada.")
        else:
            r.erro(f"texto_falado_limpo: {n} linha(s) ainda contêm tags ASR ([...] ou (...)).")

    # texto_falado_limpo e texto_falado_limpo_en devem ser lowercase
    for col in ["texto_falado_limpo", "texto_falado_limpo_en"]:
        if col not in df.columns:
            continue
        serie   = df[col].fillna("").astype(str)
        nao_lower = serie[serie.str.strip() != ""].apply(lambda x: x != x.lower()).sum()
        if nao_lower == 0:
            r.ok(f"{col}: 100% em lowercase.")
        else:
            r.erro(f"{col}: {nao_lower} linha(s) com letras maiúsculas (esperado lowercase).")

    # clickbait_score: entre 0 e 1
    if "clickbait_score" in df.columns:
        cs    = pd.to_numeric(df["clickbait_score"], errors="coerce")
        fora  = ((cs < 0) | (cs > 1)).sum()
        if fora == 0:
            r.ok("clickbait_score: todos entre 0 e 1.")
        else:
            r.erro(f"clickbait_score: {fora} valor(es) fora de [0,1].")

    # tem_repeticao_roteiro: deve ser bool ou 0/1
    if "tem_repeticao_roteiro" in df.columns:
        valores = df["tem_repeticao_roteiro"].dropna().unique()
        invalidos = [v for v in valores if v not in (True, False, 0, 1, "True", "False", "true", "false")]
        if not invalidos:
            r.ok("tem_repeticao_roteiro: apenas valores booleanos.")
        else:
            r.erro(f"tem_repeticao_roteiro: valores inesperados → {invalidos}")

    # colunas _en não devem conter letras maiúsculas onde lowercase é obrigatório
    for col in ["palavras_chave_en", "pistas_audio_en"]:
        if col not in df.columns:
            continue
        serie     = df[col].fillna("").astype(str)
        nao_lower = serie[serie.str.strip() != ""].apply(lambda x: x != x.lower()).sum()
        if nao_lower == 0:
            r.ok(f"{col}: 100% em lowercase.")
        else:
            r.erro(f"{col}: {nao_lower} linha(s) com maiúsculas (esperado lowercase).")

    # originais não devem ter sido alteradas — verifica se video_id é único
    if "video_id" in df.columns:
        n_dup = df["video_id"].duplicated().sum()
        if n_dup == 0:
            r.ok("video_id: todos únicos, originais preservados.")
        else:
            r.erro(f"video_id: {n_dup} duplicata(s) — pode indicar problema no CSV.")


def check_colunas_originais_intactas(df: pd.DataFrame, r: Relatorio):
    """Verifica se as colunas originais não foram corrompidas (tipos e completude)."""
    r.secao("7. INTEGRIDADE DAS COLUNAS ORIGINAIS")

    # Numéricas que devem ser numéricas
    numericas = ["visualizacoes", "curtidas", "comentarios", "duracao_segundos"]
    for col in numericas:
        if col not in df.columns:
            continue
        nao_num = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if nao_num == 0:
            r.ok(f"{col}: tipo numérico OK.")
        else:
            r.aviso(f"{col}: {nao_num} valor(es) não numérico(s).")

    # video_id, canal_id, thumb_maxres: não podem ser nulos
    for col in ["video_id", "canal_id", "thumb_maxres"]:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n == 0:
            r.ok(f"{col}: sem nulos.")
        else:
            r.erro(f"{col}: {n} nulo(s) — coluna original não deveria ter nulos.")

    # nicho: deve ser coluna única (todos iguais)
    if "nicho" in df.columns:
        unicos = df["nicho"].dropna().unique()
        if len(unicos) == 1:
            r.ok(f"nicho: valor único '{unicos[0]}' em todas as linhas.")
        else:
            r.aviso(f"nicho: múltiplos valores → {unicos}. Esperado apenas 1 para CSV de nicho único.")


def check_resumo_final(df: pd.DataFrame, r: Relatorio):
    """Resumo geral do dataset."""
    r.secao("8. RESUMO DO DATASET")
    r.ok(f"Total de linhas  : {len(df):,}")
    r.ok(f"Total de colunas : {len(df.columns)}")
    r.ok(f"Esperado         : {len(ORDEM_FINAL_ESPERADA)} colunas ({len(COLUNAS_ORIGINAIS)} originais + {len(ORDEM_FEATURES_CRIADAS)} criadas)")

    # Distribuições rápidas das categóricas
    for col in ["faixa_duracao", "estrutura_blocos", "label_viral", "sentimento_roteiro"]:
        if col in df.columns:
            dist = df[col].value_counts().to_dict()
            r.ok(f"{col}: {dist}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auditoria pós-ajuste do CSV processado.")
    parser.add_argument("--csv", default=CSV_DEFAULT, help="Caminho do CSV a auditar.")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  VERIFICAR_AJUSTE — Auditoria pós-00_ajuste.py")
    print(f"  CSV: {args.csv}")
    print(f"  Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    if not os.path.exists(args.csv):
        print(f"❌ Arquivo não encontrado: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv, low_memory=False)
    print(f"CSV carregado: {len(df):,} linhas × {len(df.columns)} colunas\n")

    r = Relatorio()

    check_colunas_presentes(df, r)
    check_ordem_colunas(df, r)
    check_nulos(df, r)
    check_dominios(df, r)
    check_limites_numericos(df, r)
    check_consistencia_interna(df, r)
    check_colunas_originais_intactas(df, r)
    check_resumo_final(df, r)

    # ── Placar final ──────────────────────────────────────────
    sep = "=" * 60
    placar = (
        f"\n{sep}\n"
        f"  RESULTADO FINAL\n"
        f"{sep}\n"
        f"  ✅ OK     : {r.oks}\n"
        f"  ⚠️  Avisos : {r.avisos}\n"
        f"  ❌ Erros  : {r.erros}\n"
        f"{sep}"
    )
    r.linhas.append(placar)
    print(placar)

    saida = os.path.join(os.path.dirname(args.csv), "verificar_ajuste_relatorio.txt")
    r.salvar(saida)

    sys.exit(0 if r.erros == 0 else 1)


if __name__ == "__main__":
    main()