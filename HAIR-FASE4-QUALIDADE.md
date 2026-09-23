# HAIR — Fase 4: especificação do proxy de qualidade e de "custo por sucesso"

Data: 23/09/2026 · autor: Hermes · status: **vigente, bloqueia promoção de política**

## Por que este documento existe

O plano da Fase 0 (`CURRENT_AI_ARCHITECTURE.md` §7) diz: *"Qualidade subjetiva não
deve virar fallback implícito sem uma especificação e avaliação separadas."* Sem
este documento, a Fase 4 consegue medir **custo** e **latência**, mas não tem como
provar que uma rota mais barata **não degradou** o trabalho — e portanto não pode
autorizar política mais agressiva. Este é o artefato que faltava para fechar o
projeto de engenharia de tokens.

## Princípio

Nenhuma política de rota é promovida sem os quatro itens juntos:

1. **Proxy medível** de qualidade (não opinião) — definido abaixo.
2. **Amostra mínima por braço** (controle × candidato) na **mesma** configuração.
3. **Comparação antes/depois** com o mesmo mix de tarefas.
4. **Rollback declarado** antes de ligar (flag off / `git checkout` do commit).

## Proxies v1 — medidos do `state.db`, sem instrumentação nova

Todos saem de tabelas que **já existem** (`messages`, `sessions`,
`session_model_usage`). Nenhum deles grava texto de usuário em telemetria nova.

| # | Proxy | Definição | Fonte |
|---|-------|-----------|-------|
| **P1** | `taxa_falha_tool` | tools com `exit_code <> 0` **ou** `error` não-vazio ÷ total de tools | `messages.role='tool'`, JSON |
| **P2** | `taxa_truncamento` | `finish_reason='length'` ÷ mensagens de assistant | `messages.finish_reason` |
| **P3** | `turnos_por_sessao` | média de `message_count` por sessão (eficiência: mais turnos = mais retrabalho) | `sessions.message_count` |
| **P4** | `fim_anormal` | sessões com `end_reason` fora de `{cron_complete, agent_close}` | `sessions.end_reason` |
| **P5** | `custo_por_sessao_limpa` | USD total ÷ sessões com `P1 = 0` — **v1 de custo por sucesso** | `session_model_usage` × `sessions` |
| **P6** | `friccao_guardrail` | ocorrências de `BLOCKED` em resultado de tool (payload recusado) | `messages.content` |

SQL de referência (P1, o central):

```sql
SELECT json_extract(m.content,'$.exit_code') <> 0
    OR json_extract(m.content,'$.error')   <> ''
FROM messages m WHERE m.role = 'tool'
```

## Baseline medida (23/09/2026, janela de 7 dias)

```
modelo                       | sessoes | tools | falhas | %falha | msgs/sessao
deepseek-v4-flash            |     208 |  8695 |    354 |  4.07% |        54.1
deepseek-flash               |       6 |   404 |     13 |  3.22% |       117.8
gpt-5.6-sol                  |      10 |   241 |      8 |  3.32% |        42.9
nex-agi/nex-n2.5-mini:free   |      14 |   220 |     15 |  6.82% |        23.3
qwen3.5-4b-local             |       3 |     0 |      0 |      - |         2.0
truncamento: 0 · fim de sessao: cron_complete=213, agent_close=12, (nulo)=11, session_reset=9
```

## Achado que já muda a decisão (leia antes de qualquer política)

**O tier `local` (Qwen 3.5 4B) não tem uma única chamada de tool em 7 dias** — 3
sessões, média de **2 mensagens**. Ou seja: o tier "econômico" até agora só
conversou curto. Não existe evidência de que ele **execute trabalho real** (arquivo,
teste, terminal). Enquanto isso não existir, qualquer política que empurre trabalho
para o local é **aposta**, não economia — e a comparação de `%falha` entre modelos
é confundida pelo mix de tarefas (o `nex:free` com 6,82% provavelmente recebe
tarefas diferentes do `deepseek`).

## O que ainda NÃO é medível (instrumentação pendente, v2)

1. **"A tarefa foi cumprida?"** — hoje só se infere por ausência de erro. Um proxy
   melhor exige um sinal de conclusão por tarefa (ex.: verificação posterior) — é
   o item mais caro e mais valioso.
2. **Atribuição por turno** — o log do roteador é por sessão/tier; sessão que
   escala troca de modelo no meio, e hoje isso não é separado na conta.
3. **Rejeição/refação do usuário** — medível por padrão lexical em `messages`
   ("não é isso", "refaz", "de novo", "errado"), com lista curada; proposta v2.
4. **Custo de auxiliares e compressão por sessão** — existe em
   `session_model_usage` por `task`, mas ainda não entra no custo por sessão.

## Regra de promoção (gate)

Uma política só é promovida quando **todos** valem, por braço:

- **n ≥ 30 sessões** e **≥ 7 dias** na mesma configuração, com **mix de tarefas comparável**;
- **P1** não sobe mais de **1 ponto percentual** absoluto vs. controle;
- **P3** não sobe mais de **20%** vs. controle;
- **P2 = 0** (nenhum truncamento novo);
- **P4** sem tipo novo de `end_reason` anômalo;
- **P5 cai** — a promoção é por *custo por sucesso*, nunca por custo por token;
- rollback declarado e testado antes de ligar.

Qualquer item violado = **mantém como está** e o achado vai para o relatório.

## Como rodar

```bash
python "$LOCALAPPDATA/hermes/hermes-agent/venv/Scripts/python.exe" \
       "$LOCALAPPDATA/hermes/scripts/hair_evidence.py"
```

A seção `qualidade` do coletor imprime P1/P3/P2/P4 na janela pedida (`--days`).
