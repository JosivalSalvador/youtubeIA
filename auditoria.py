import os
import pandas as pd

# Importações do seu projeto
from conexoes_externas import baixar_csv_mestre, NOME_ARQUIVO_MESTRE

def auditoria_megazord_v3():
    print("="*75)
    print("🚀 INICIANDO AUDITORIA GLOBAL: MEGAZORD ANALÍTICO V3 (ESTADO REAL DOS DADOS) 🚀")
    print("="*75)

    print("\n📥 Puxando a montanha de dados do Supabase...")
    try:
        df = baixar_csv_mestre()
    except Exception as e:
        print(f"❌ Erro ao baixar o arquivo: {e}")
        return

    # Tratamento global e conversão numérica
    df = df.fillna("")
    colunas_numericas = ['score_viral', 'velocidade_views', 'ritmo_palavras_seg', 'clickbait_score', 'taxa_conversao', 'duracao_segundos']
    for col in colunas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"\n📦 VOLUME TOTAL: {len(df)} vídeos | 🧩 VARIÁVEIS: {len(df.columns)}")
    if len(df.columns) == 53:
        print("   ✅ Estrutura perfeita de 53 colunas confirmada.")

    # ======================================================================
    print("\n" + "="*75)
    print("🔬 MÓDULO 1: DIAGNÓSTICO DE INTEGRIDADE E IDENTIFICAÇÃO DE FALHAS")
    print("="*75)
    
    falhas_raspagem = df[df['texto_falado'].astype(str).str.contains(r'\[ERRO', na=False, regex=True)]
    videos_mudos_df = df[(df['texto_falado'] == "") | (df['texto_falado'].isna())]
    falhas_gemini = df[(df['descricao_visual_thumb'] == "") | (df['descricao_visual_thumb'].isna())]

    print(f"   -> Erros de Raspagem de Legenda: {len(falhas_raspagem)}")
    if not falhas_raspagem.empty:
        print("      ⚠️ DETALHAMENTO DAS FALHAS DE TEXTO:")
        for _, row in falhas_raspagem.iterrows():
            print(f"      🚫 ID: {row['video_id']} | Nicho: {row['nicho']} | Título: {row['titulo'][:40]}...")

    # AQUI ESTÁ A RESPOSTA SOBRE OS VÍDEOS MUDOS
    print(f"\n   -> Vídeos 'Mudos' (Sem fala/Só Música confirmados): {len(videos_mudos_df)}")
    if not videos_mudos_df.empty:
        print("      🔎 ORIGEM DOS VÍDEOS MUDOS (Faz sentido não ter voz?):")
        mudos_por_nicho = videos_mudos_df['nicho'].value_counts()
        for nicho, count in mudos_por_nicho.items():
            print(f"      - {nicho}: {count} vídeos mudos")
    
    print(f"\n   -> Falhas Críticas na Visão Computacional (Gemini engasgou): {len(falhas_gemini)}")
    if not falhas_gemini.empty:
        print("      ⚠️ DETALHAMENTO DOS ENGASGOS DA IA VISUAL:")
        for _, row in falhas_gemini.iterrows():
             print(f"      - ID: {row['video_id']} | Nicho: {row['nicho']}")
             print(f"        Thumb URL: {row['thumb_maxres']}")
             
    if len(df) >= 299:
        df_originais = df.head(299)
        df_novos = df.tail(len(df) - 299)
        print(f"\n   -> Lote Original Mestre: 299 vídeos (Corrompidos: {df_originais[df_originais['video_id'] == ''].shape[0]})")
        print(f"   -> Nova Fornada Inserida: {len(df_novos)} vídeos")
    
    nichos_acima_teto = len(df.groupby('nicho').filter(lambda x: len(x) > 200)['nicho'].unique())
    if nichos_acima_teto == 0:
        print("   ✅ Guilhotina Matemática: Nenhum nicho ultrapassou 200 vídeos no total.")
    else:
        print(f"   ⚠️ ALERTA: {nichos_acima_teto} nicho(s) ultrapassaram a marca de 200 vídeos.")

    # ======================================================================
    print("\n" + "="*75)
    print("🏆 MÓDULO 2: DESEMPENHO CRUZADO POR NICHO (RANKING DETALHADO)")
    print("="*75)
    
    if 'nicho' in df.columns and 'score_viral' in df.columns:
        nicho_stats = df.groupby('nicho').agg(
            qtd=('video_id', 'count'),
            avg_score=('score_viral', 'mean'),
            max_score=('score_viral', 'max'),
            avg_vel=('velocidade_views', 'mean'),
            avg_conv=('taxa_conversao', 'mean')
        ).sort_values(by='avg_score', ascending=False)
        
        for nicho, row in nicho_stats.iterrows():
            print(f"   -> {nicho.upper()} ({row['qtd']} vídeos):")
            print(f"      🔥 Média Score: {row['avg_score']:,.0f} | 👑 Teto (Máx): {row['max_score']:,.0f}")
            print(f"      🚀 Vel: {row['avg_vel']:,.0f} views/dia | 🧲 Conversão: {row['avg_conv']:.2f}%")

    # ======================================================================
    print("\n" + "="*75)
    print("👑 MÓDULO 3: OS EXTREMOS DA BASE (TOP 3 E BOTTOM 3)")
    print("="*75)
    
    if 'score_viral' in df.columns:
        df_sorted = df.sort_values(by='score_viral', ascending=False)
        top_3 = df_sorted.head(3)
        bottom_3 = df_sorted.tail(3)
        
        print("   🥇 O PÓDIO (Maior Retenção Algorítmica):")
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f"      {i}º) ID: {row['video_id']} | Score: {row['score_viral']:,.0f} | Conv: {row['taxa_conversao']:.2f}% | Nicho: {row['nicho']}")
            
        print("\n   🐢 O CEMITÉRIO (Piores Desempenhos):")
        for i, (_, row) in enumerate(bottom_3.iterrows(), 1):
            print(f"      -{i}º) ID: {row['video_id']} | Score: {row['score_viral']:,.0f} | Conv: {row['taxa_conversao']:.2f}% | Nicho: {row['nicho']}")

    # ======================================================================
    print("\n" + "="*75)
    print("🧠 MÓDULO 4: PADRÕES DE COPY, EMOJIS E ROTEIRO")
    print("="*75)
    
    if 'estrutura_blocos' in df.columns:
        total_com_estrutura = len(df[df['estrutura_blocos'] != ""])
        print(f"   -> Top 5 Estruturas de Roteiro mais tracionadas (Base: {total_com_estrutura}):")
        top_estruturas = df[df['estrutura_blocos'] != ""].value_counts('estrutura_blocos').head(5)
        for est, count in top_estruturas.items():
            pct = (count / total_com_estrutura) * 100
            print(f"      - {est}: {count} vídeos ({pct:.1f}%)")
            
    if 'vibe_emojis' in df.columns:
        print("\n   -> Top 5 Combos de Emojis dominantes:")
        top_emojis = df[df['vibe_emojis'] != ""].value_counts('vibe_emojis').head(5)
        for emo, count in top_emojis.items():
            print(f"      - {emo} : {count} vídeos")

    # ======================================================================
    print("\n" + "="*75)
    print("🔗 MÓDULO 5: MATRIZ DE RETENÇÃO E CORRELAÇÃO")
    print("="*75)
    
    if all(c in df.columns for c in colunas_numericas):
        corr_matrix = df[colunas_numericas].corr()
        print("   -> O que mais impacta o SCORE VIRAL? (-1.0 a 1.0)")
        
        def classificar_corr(val):
            if val > 0.5: 
                return "🟢 Forte Positiva"
            if val > 0.1: 
                return "🟡 Fraca Positiva"
            if val > -0.1: 
                return "⚪ Neutra"
            if val > -0.5: 
                return "🟠 Fraca Negativa"
            return "🔴 Forte Negativa"

        vel_corr = corr_matrix.loc['score_viral', 'velocidade_views']
        ritmo_corr = corr_matrix.loc['score_viral', 'ritmo_palavras_seg']
        click_corr = corr_matrix.loc['score_viral', 'clickbait_score']
        dur_corr = corr_matrix.loc['score_viral', 'duracao_segundos']

        print(f"      📈 Velocidade de Views: {vel_corr:.3f} ({classificar_corr(vel_corr)})")
        print(f"      🗣️ Ritmo (Palavras/seg): {ritmo_corr:.3f} ({classificar_corr(ritmo_corr)})")
        print(f"      🧲 Apelo (Clickbait Score): {click_corr:.3f} ({classificar_corr(click_corr)})")
        print(f"      ⏱️ Duração do Vídeo: {dur_corr:.3f} ({classificar_corr(dur_corr)})")

    # ======================================================================
    print("\n" + "="*75)
    print("⏰ MÓDULO 6: O RELÓGIO VIRAL E LIMITES DE FORMATO")
    print("="*75)
    
    if 'dia_postagem' in df.columns and 'hora_postagem' in df.columns:
        melhor_dia = df.groupby('dia_postagem')['score_viral'].mean().sort_values(ascending=False).head(3)
        melhor_hora = df.groupby('hora_postagem')['score_viral'].mean().sort_values(ascending=False).head(3)
        
        print("   -> 📅 Top 3 Dias da Semana:")
        for d, v in melhor_dia.items(): 
            print(f"      - {d}: {v:,.0f} pts")
            
        print("\n   -> 🕒 Top 3 Horários (UTC):")
        for h, v in melhor_hora.items(): 
            print(f"      - {int(h)}h: {v:,.0f} pts")

    if 'duracao_segundos' in df.columns:
        _, bins = pd.qcut(df['duracao_segundos'].dropna(), q=3, retbins=True, duplicates='drop')
        if len(bins) >= 4:
            print(f"\n   -> 🎬 Qual tamanho retém mais? (Cortes reais da base: <{bins[1]:.0f}s | {bins[1]:.0f}-{bins[2]:.0f}s | >{bins[2]:.0f}s)")
            df['faixa_tempo'] = pd.qcut(df['duracao_segundos'].dropna(), q=3, labels=['Curtos', 'Médios', 'Longos'], duplicates='drop')
            faixas = df.groupby('faixa_tempo', observed=False)['score_viral'].mean().sort_values(ascending=False)
            for faixa, media in faixas.items():
                print(f"      - {faixa}: {media:,.0f} pts")

    # ======================================================================
    print("\n" + "="*75)
    print("📊 MÓDULO 8: ESTADO REAL DAS 53 VARIÁVEIS POR NICHO (MAPA DE BURACOS)")
    print("="*75)
    
    # Itera sobre todos os nichos e verifica todas as 53 colunas
    for nicho in df['nicho'].unique():
        df_nicho = df[df['nicho'] == nicho]
        total_nicho = len(df_nicho)
        print(f"\n   -> 📌 {nicho.upper()} (Total: {total_nicho} vídeos):")
        
        colunas_com_furo = 0
        for col in df.columns:
            # Conta nulos e strings vazias/só com espaços
            furos = df_nicho[col].isna() | (df_nicho[col].astype(str).str.strip() == "")
            qtd_furos = furos.sum()
            
            if qtd_furos > 0:
                colunas_com_furo += 1
                pct_furo = (qtd_furos / total_nicho) * 100
                print(f"      ⚠️ {col}: {qtd_furos} vazios ({pct_furo:.1f}% de buraco)")
                
        if colunas_com_furo == 0:
            print("      ✅ Perfeito: Todas as 53 variáveis estão 100% preenchidas neste nicho.")

    # ======================================================================
    print("\n" + "="*75)
    print("🧹 MÓDULO 7: LIMPANDO O CHÃO DE FÁBRICA")
    print("="*75)
    
    if os.path.exists(NOME_ARQUIVO_MESTRE):
        os.remove(NOME_ARQUIVO_MESTRE)
        print("   ✅ Arquivo CSV deletado fisicamente do HD. Sem rastros.")

    print("\n=== 🏁 AUDITORIA MEGAZORD V3 FINALIZADA COM SUCESSO 🏁 ===")

if __name__ == "__main__":
    auditoria_megazord_v3()