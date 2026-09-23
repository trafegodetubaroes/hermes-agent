# Arquitetura atual de IA — auditoria Phase 0

**Escopo:** fotografia do repositório e dos fatos de ambiente fornecidos para esta fase. Este arquivo é documentação; não propõe ativação de comportamento em produção. Segredos, chaves, tokens e valores de `.env`/`auth.json` não foram lidos nem reproduzidos.

## 1. Estado atual e fluxo de dados

O Hermes tem um núcleo conversacional compartilhado por CLI, gateway, TUI, cron e delegação. O caminho nominal é:

```text
entrada (CLI/gateway/TUI/cron)
  → resolução de sessão, modelo e runtime
  → AIAgent (run_agent.py)
  → loop de mensagens + ferramentas
  → resolve_runtime_provider()
  → API compatível / Codex Responses / Anthropic Messages / endpoint customizado
  → normalização de uso, persistência e resposta
```

- `hermes_cli/cli_agent_setup_mixin.py::_init_agent` constrói o agente da CLI; `run_agent.py::AIAgent` contém o ciclo principal e aceita `model`, `provider`, `fallback_model`, `IterationBudget` e contexto de sessão.
- `hermes_cli/runtime_provider.py::resolve_runtime_provider` transforma a escolha lógica em credenciais, URL e `api_mode`; `hermes_cli/providers.py::ProviderDef` e o catálogo de provedores fornecem metadados e normalização.
- `agent/chat_completion_helpers.py` prepara chamadas, retries e ativação de fallback. `run_agent.py::AIAgent._try_activate_fallback` é o ponto de entrada do agente.
- Ferramentas retornam ao loop como mensagens `tool`; delegações criam agentes filhos em `tools/delegate_tool.py`, que podem herdar cadeia de fallback e contexto permitido.
- Auxiliares (visão, compressão, título, web extract e similares) são roteados por `agent/auxiliary_client.py`, separados do loop principal e identificados por `task` na contabilidade.
- Compressão/context engine usa `agent/context_compressor.py`, `agent/context_engine.py` e carregamento de `plugins/context_engine/`; altera o contexto apenas na compressão. Isso é compatível com o princípio de que o cache de prompt por conversa é sagrado.
- Hooks de ciclo de vida são registrados em `hermes_cli/hooks.py` e podem incluir `pre_llm_call`/`subagent_stop`; plugins são carregados nas interfaces existentes, não no esquema central de ferramentas.

## 2. Inventário exato observado

### Rota principal atual

| Item | Estado verificado |
|---|---|
| Provider principal | `openai-codex` |
| Modelo principal | `gpt-5.6-sol` |
| Credenciais diretas configuradas | DeepSeek API, Gemini API, OpenRouter API, OpenAI Codex OAuth e endpoint local Qwen customizado |
| Credenciais diretas ausentes | Não há chave direta Anthropic nem chave direta OpenAI |
| Endpoint local | `http://127.0.0.1:8081/v1` |
| Modelo local | `qwen3.5-4b-local`, GGUF Q6_K, 4.205B parâmetros, `n_ctx=65536`; o endpoint reporta multimodalidade |

Verificação em runtime do endpoint local: `llama-server.exe` (PID 51692, ~3,6 GB de memória) escutando em `127.0.0.1:8081`, respondendo HTTP 200 em `/v1/models`. **Não há tarefa agendada nem entrada de startup que o relance** — foi iniciado manualmente, então uma reinicialização do Windows elimina o Tier 0 e qualquer lane econômico que dependa dele. Isso é um gap de disponibilidade, não um bug de código, e precisa ser resolvido antes de promover lanes locais.

### Cache do Codex CLI

Modelos disponíveis no cache: `gpt-6-astra`, `gpt-reserve`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5` e `codex-auto-review`. O default do CLI Codex é `gpt-6-astra`; isso não substitui o modelo principal atual do Hermes (`gpt-5.6-sol`).

### Rotas auxiliares e especiais

| Tarefa/uso | Rota configurada ou observada |
|---|---|
| Delegação | `deepseek/deepseek-flash` |
| Visão | `openrouter/google/gemini-2.5-flash-lite` |
| Muitos auxiliares baratos | `custom:local-qwen/qwen3.5-4b-local`, com fallback NEX free e depois GPT |
| Cron pins observados | `deepseek/deepseek-v4-flash` e `openrouter/nex-agi/nex-n2.5-mini:free` |

Este inventário é a configuração/ambiente auditado, não uma afirmação de que os demais `ProviderConfig` registrados em `hermes_cli/auth.py::PROVIDER_REGISTRY` estejam autenticados ou disponíveis. O catálogo também suporta provedores adicionais e providers customizados via `hermes_cli/providers.py`; disponibilidade efetiva passa por `resolve_runtime_provider`.

## 3. Pontos de seleção e precedência

### CLI

`hermes_cli/cli_agent_setup_mixin.py::_init_agent` combina override de invocação/sessão e configuração antes de construir `AIAgent`. A seleção conceitual é:

1. override explícito do comando/sessão, incluindo `--model` e provider explícito;
2. modelo/provider persistidos na sessão ou escolhidos por `/model`/`switch_model`;
3. `model.default`/`model.model` e `model.provider` em `config.yaml`;
4. descoberta automática de credenciais por `resolve_requested_provider` e `resolve_runtime_provider` quando provider é `auto`/implícito.

`hermes_cli/model_switch.py::switch_model` e `run_agent.py::AIAgent.switch_model` mudam o runtime corrente. Uma troca no meio da conversa é operacionalmente cara: pode exigir preflight/compressão e romper o prefixo cacheável. Portanto, qualquer roteador futuro deve decidir uma vez e manter a sessão sticky.

### Gateway/TUI

O gateway mantém override por conversa/sessão. Em `gateway/run.py`, `_apply_session_model_override` aplica o override de sessão antes da criação do agente; o estado é reidratado e persistido. Em `tui_gateway/server.py`, `model_override` é propagado por sessão até `AIAgent` (`_make_agent`, definido na linha 6489, recebe `model_override` na linha 6494). O override explícito de sessão vence configuração global; na ausência dele, aplica-se a resolução global do runtime. O código evita usar variáveis de ambiente de processo como estado de GUI/sessão.

### Cron

`cron/scheduler.py` documenta e implementa a precedência: **override por job > `cron.model`/`cron.model_provider` > `HERMES_MODEL` > `model.default`**; o provider é resolvido depois por `resolve_runtime_provider`. `cron.model` é o default deliberado da frota, distinto do provider do scheduler. O guard de deriva é `cron_model_drift_guard_enabled()` em `hermes_cli/config.py` (linha 4581), cujo default `model_drift_guard: True` vive em `hermes_cli/config_defaults.py` (linha 2253); ele protege jobs não pinados contra deriva silenciosa. Fallback do agente é entregue via `get_fallback_chain`.

### Delegação

`tools/delegate_tool.py::_build_child_agent` e `_resolve_delegation_credentials` dão precedência a `delegation.base_url`/credencial explícita, depois a `delegation.provider` + `delegation.model`; se não houver override, o filho herda o runtime permitido do pai/configuração. A configuração auditada fixa `deepseek/deepseek-flash`. A cadeia de fallback do pai é propagada, mas a delegação não é um roteamento subjetivo de qualidade.

### Auxiliares

`agent/auxiliary_client.py` resolve por tarefa, provider e modelo configurados; `_resolve_auto`/`_resolve_auto_route` e `ProviderProfile.resolve_aux_model` participam da escolha. A rota pode ser diferente do agente principal: visão está em OpenRouter/Gemini e muitos auxiliares em Qwen local, com fallback NEX free e GPT. A escolha de auxiliar não deve alterar o modelo sticky do loop principal.

### Fallback

`hermes_cli/fallback_config.py::get_fallback_chain` combina `fallback_providers` em ordem e depois o legado `fallback_model`. `agent/chat_completion_helpers.py::try_activate_fallback` chama `switch_model`/restauração quando há falha de provider/runtime, timeout, rate limit ou falha operacional equivalente. O fallback existente **não** escala por avaliação subjetiva, teste de qualidade ou classificação semântica; Phase 1+ não deve afirmar que essa capacidade já existe.

## 4. Contabilidade de tokens e custo

`agent/usage_pricing.py::normalize_usage` normaliza input, output, cache read/write e reasoning conforme o wire protocol; `estimate_usage_cost` resolve rota de billing e tabela de preços. `hermes_state.py::SessionDB::_record_model_usage` grava em `session_model_usage` por `(session_id, model, billing_provider, billing_base_url, billing_mode, task)`, incluindo contagem de chamadas, categorias de tokens, custos estimado/real, status, source, versão/preços e timestamps. `record_auxiliary_usage` mantém auxiliares fora do resumo principal para não duplicar contadores.

Baseline de 30 dias do Hermes State DB:

| Métrica | Valor |
|---|---:|
| Sessões | 742 |
| Mensagens | 27.277 |
| Chamadas de ferramenta | 13.825 |
| Input | 70.829.962 |
| Output | 17.422.256 |
| Total incluindo cache/reasoning | 2.924.495.568 |
| Main-agent API calls | 22.664 |
| Main input/output | 69.996.222 / 16.173.211 |
| Main cache-read | 2.837.028.005 |
| Main custo estimado | USD 39,013066 |
| Auxiliares | visão: 424 / USD 0,133076; compressão: 26 / USD 0,040388; título: 18 / USD 0,000528 |
| Total estimado (loop principal + auxiliares contabilizados) | USD 39,187058 |

Leitura operacional do baseline: custo real é majoritariamente zero/desconhecido; registros Codex subscription-included são estimados/rotulados conforme a rota. O maior componente estimado é DeepSeek V4 Flash, aproximadamente USD 36,62 nos modos de billing observados (USD 31,190408 em billing normal + USD 5,426107 em billing `subscription_included`); houve 1.746 chamadas GPT-5.6 Sol com custo estimado zero por estarem incluídas na assinatura, e somente 5 chamadas contabilizadas do Qwen local. Esses números são um baseline, não uma promessa de cobrança do provedor.

Gaps atuais: campos reais de cobrança frequentemente não retornam; cache/reasoning variam por provider e wire format; auxiliares podem ter metadados incompletos; chamadas de fallback e compressão exigem correlação por task/model/provider para atribuição perfeita; `usage_totals` resume sessões principais, enquanto analytics precisa unir `session_model_usage`. Não há ainda um custo marginal confiável para cada decisão de roteamento nem medição causal de qualidade por rota.

## 5. Oportunidades de economia

- Roteamento determinístico por classe: manter tarefas simples em Qwen local ou NEX free, reservar DeepSeek/GPT para raciocínio, ferramentas difíceis e recuperação.
- Continuar separando auxiliares do loop principal; visão e compressão são candidatos a limites de tokens, cache de resultados e modelos locais compatíveis.
- Fixar cron e delegação em lanes econômicos quando a tarefa permitir, sem herdar involuntariamente o modelo interativo pago.
- Usar `session_model_usage` como fonte de custo por task/model/provider e medir custo por sucesso, não apenas custo bruto.
- Preservar cache de prompt: não trocar modelo por turno; economias que invalidem prefixos podem aumentar o custo total.
- Carregar preços/IDs do catálogo e da configuração, nunca hardcodar nomes, preços ou segredos no roteador.

## 6. Riscos de migração

1. Trocar modelo durante uma sessão invalida economia de cache, altera contexto e pode exigir compressão; o risco é maior em gateway/TUI persistentes.
2. Alterar precedência pode fazer cron não pinado seguir um modelo pago ou mudar após restart; manter `model_drift_guard` e pins explícitos.
3. Modelos locais podem reportar multimodalidade mas falhar em tool calling, contexto real ou latência; capability declarada não é prova de qualidade.
4. Provider IDs, endpoints customizados, OAuth e `api_mode` têm semântica própria em `resolve_runtime_provider`; duplicar essa lógica gera falhas de credencial e cobrança.
5. Fallback amplo pode escalar custo ou vazar dados para outro provider; preservar allowlists, billing mode e observabilidade sem texto de prompt.
6. Auxiliar diferente pode alterar formato de resumo/visão e degradar o agente sem aparecer como falha de runtime.
7. Delegação concorrente e cron tornam decisões processuais compartilhadas perigosas; nunca usar estado global para uma decisão que é de sessão.

## 7. Plano de implementação proposto, adaptado aos símbolos reais

### Fase 0 — concluída neste arquivo

Documentar arquitetura, inventário, precedência, baseline, riscos e invariantes. Nenhuma mudança de produção.

### Fase 1 — shadow router opt-in

1. Adicionar um módulo estreito em `agent/` ou `hermes_cli/` que receba somente metadados já resolvidos: superfície, sessão, provider/model explícitos, capacidade necessária, budget e classe determinística.
2. Integrá-lo como observador próximo da criação em `cli_agent_setup_mixin.py::_init_agent`, gateway antes da criação do agente em `gateway/run.py`, TUI antes de `tui_gateway/server.py::_make_agent`, cron após a precedência de `cron/scheduler.py`, e delegação antes de `tools/delegate_tool.py::_build_child_agent`.
3. Em modo shadow, chamar `resolve_runtime_provider` apenas para validar a rota candidata, sem alterar `AIAgent`, `model_override`, `fallback_model` ou `auxiliary_client`.
4. Emitir telemetria limitada a IDs de rota, classe, decisão, motivo, latência, tokens/custo já disponíveis e status; nunca prompt, mensagem, ferramenta, URL com segredo ou conteúdo de usuário.
5. Persistir resultados em estrutura compatível com `session_model_usage` ou tabela/versionamento separado, usando IDs e preços vindos da configuração/catálogo.
6. Avaliar custo, latência, erro operacional, cache-read e proxies de qualidade já existentes. Não introduzir uma chamada LLM extra.

### Fase 2 — rotas econômicas explicitamente pinadas

Depois de avaliação, habilitar somente para novas sessões e superfícies selecionadas. O roteador escolhe uma vez por sessão; pins explícitos de usuário, sessão, cron e delegação sempre vencem. Auxiliares continuam independentes. Fallback permanece operacional, não qualitativo.

### Fase 3 — expansão controlada

> **Implementada em 2026-09-23** (branch `feature/hair-phase0`, router
> `phase3-1`, 39 testes novos em `tests/agent/test_adaptive_routing_phase3.py`).
> Entregue como gates opt-in em `agent.adaptive_routing`, todos com default que
> preserva o comportamento da Phase 2 — nada muda até serem ligados:
> `surfaces` (allowlist de quem pode APLICAR; só cli+gateway por default,
> fail-closed para superfícies desconhecidas), `budget` (teto diário de rotas
> aplicadas por tier, contado do próprio log; tier gasto faz o roteador descer
> a escada, nunca subir) e `data_policy` (`local_only_classes` — classes que
> não saem da máquina; `forbidden_tiers` — tiers nunca selecionados nem
> escalados). Um gate negado registra a decisão como `applied=false` com o
> motivo, para a Phase 4 decidir com evidência em vez de suposição.

Adicionar allowlists por capacidade, budgets e política de dados; testar CLI, gateway, TUI, cron, delegação e auxiliares com `HERMES_HOME` temporário e imports reais. Medir invariantes: alternância de mensagens, cache do prefixo, billing attribution, isolamento de sessão e restauração de fallback.

### Fase 4 — otimização baseada em evidência

Só após dados suficientes considerar políticas de custo/latência mais agressivas, mantendo decisão sticky, configuração declarativa e rollback por sessão. Qualidade subjetiva não deve virar fallback implícito sem uma especificação e avaliação separadas.

## 8. Rollback e shadow mode

- Feature flag opt-in em `config.yaml`, desligada por default; nenhum novo `HERMES_*` para configuração comportamental.
- Shadow mode calcula e registra a decisão candidata, mas o caminho efetivo continua sendo o existente. Não troca provider/model, não altera prompt e não injeta chamada LLM.
- Guardar `router_version`, `decision_id`, rota efetiva e rota candidata; telemetria bounded, amostrada e sem texto de usuário.
- Rollback imediato: desligar a flag, ignorar decisões shadow e continuar usando `resolve_runtime_provider`, `get_fallback_chain`, `switch_model` e `auxiliary_client` atuais. Não reescrever histórico nem apagar contabilidade.
- Ativar por superfície/sessão e excluir cron, delegação ou modelos locais se houver erro, custo inesperado, degradação de tool calling ou perda de cache.
- O gate de promoção deve exigir ausência de aumento material de falhas, confirmação de atribuição de custo e preservação do cache; sem isso, permanecer shadow.

## 9. Checklist de conclusão da Phase 0

- [x] Arquitetura/data flow auditados em símbolos reais.
- [x] Inventário atual de providers, modelos, endpoint local, pins e credenciais descrito sem valores secretos.
- [x] Precedência separada para CLI, gateway/TUI, cron, delegação, auxiliares e fallback.
- [x] Baseline de 30 dias e limites de `usage_pricing`/`session_model_usage` registrados.
- [x] Oportunidades, riscos, plano faseado, shadow mode e rollback definidos.
- [x] Prompt caching tratado como invariante; primeira decisão proposta é session-sticky.
- [x] Fallback descrito corretamente como recuperação operacional, sem alegar escalonamento por qualidade.
- [x] **Nenhuma mudança de comportamento de produção foi feita.**
- [x] **Somente este arquivo raiz foi criado nesta tarefa.**

## Separação explícita

### VERIFIED CURRENT STATE

É o conteúdo factual acima: símbolos e precedências existentes, rotas configuradas/fornecidas, baseline do State DB, contabilidade disponível, fallback operacional e compressão/context engine existentes.

### GAPS

São as limitações atuais: custo real ausente ou desconhecido em muitos registros, atribuição causal de qualidade/custo não disponível, heterogeneidade de usage/caching, metadados auxiliares incompletos e nenhuma decisão sticky de roteador econômico implementada.

### PROPOSED (NOT YET IMPLEMENTED)

São exclusivamente o roteador shadow opt-in, classificação determinística, telemetria bounded, promoção por fases, lanes econômicos, gates de qualidade/custo e rollback descritos nas seções 5–8. Nada disso foi ativado ou implementado nesta Phase 0.
