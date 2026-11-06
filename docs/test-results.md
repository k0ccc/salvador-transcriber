# Результаты проверок

Дата финальной проверки: 2026-11-06. Инициализирован новый Git-репозиторий без commit; commit hash отсутствует. Матрица соответствия требованиям находится в [`docs/requirements-coverage.md`](requirements-coverage.md).

## Окружение

- Python 3.14.9, изолированное окружение `.venv`.
- fastapi 0.162.0, httpx 0.28.1, pydantic 2.14.3, uvicorn 0.54.0, websockets 15.0.1.
- Автоматические тесты и fake-smoke не использовали сеть или OpenAI API key. Для изолированной сборки wheel pip установил build dependencies из package index; это не было обращением к OpenAI.

## Финальные автоматические проверки

Команды выполнялись последовательно после исправлений и удаления локальных build/smoke-артефактов.

| Команда | Фактический итоговый результат |
|---|---|
| `.venv/bin/python -m pytest -q` | **84 passed**, 1 warning; 3.36 s |
| `.venv/bin/python -m ruff check .` | **All checks passed** |
| `.venv/bin/python -m mypy` | **Success: no issues found in 38 source files**; strict mode задан в [`pyproject.toml`](../pyproject.toml) |
| `.venv/bin/python -m ruff format --check .` | **62 files already formatted** |

Единственный warning — `StarletteDeprecationWarning` из установленного `fastapi/testclient.py` о будущем переходе TestClient с `httpx`; это не test failure и не предупреждение из бизнес-логики.

Автоматический набор включает unit/integration/fake-сценарии обычного разговора, одновременной речи, пауз, reconnect и двух звонков, а также проверки контрактов, PCM/ресэмплинга, SQLite, HTTP export, bounded queues, gaps, metrics, latency, demo и live-runner guards.

### Исправленные дефекты и regression tests

Каждая из трёх новых проверок была сначала запущена на исходном коде и воспроизвела дефект; после исправления весь набор прошёл.

1. **Stop мог закрыть provider до отложенного финала без предшествующего partial.** Теперь после отправленного аудио всегда применяется configured final grace, а timeout по-прежнему аннулирует in-flight partial с gap. Регрессия: [`test_shutdown_waits_for_delayed_final_without_partial()`](../tests/unit/test_reliability.py:384).
2. **Гонка exporter shutdown оставляла уже dequeued событие в статусе pending.** Inflight событие теперь фиксируется непосредственно перед отменой worker и помечается failed/shutdown_timeout. Регрессия: [`test_export_shutdown_race_marks_newly_dequeued_delivery_failed()`](../tests/unit/test_reliability.py:564).
3. **OpenAI adapter скрывал различные повторные completed events от доменной дедупликации и conflict metrics.** Точные retransmission с тем же event ID подавляются транспортом, а различные completed events передаются в доменный dedup; первый финал остаётся авторитетным. Регрессия: [`test_distinct_duplicate_finals_reach_domain_dedup_and_conflict_metrics()`](../tests/unit/test_openai_realtime.py:246).

Для текущего конфигурационного исправления добавлен focused regression [`test_openai_model_has_safe_default_and_preserves_explicit_override()`](../tests/unit/test_config.py:4): default равен `gpt-4o-transcribe`, а явный override, включая `gpt-live-transcribe`, сохраняется.

Эти результаты подтверждают детерминированную логику и оркестрацию. Они **не являются** измерением качества или задержки настоящего OpenAI.

## Packaging и чистая установка

| Проверка | Фактический результат |
|---|---|
| `.venv/bin/python -m pip wheel . --no-deps --wheel-dir dist` | Wheel `salvador_transcriber-0.1.0-py3-none-any.whl` успешно собран |
| Установка wheel с `--no-deps --target var/package-smoke` | **Success** |
| Содержимое wheel | Присутствуют `salvador_transcriber/demo/static/index.html` и entry point `salvador-transcriber = salvador_transcriber.main:run`; каталоги `var/` и `tests/` не включены |
| Import из установленной копии | Модуль загружен именно из `var/package-smoke`, а не из editable source |
| Packaged demo | [`demo_html()`](../src/salvador_transcriber/demo/page.py:4) вернула страницу с независимыми `guestFile` и `operatorFile` |

Wheel, установленная копия, egg-info и smoke-файлы были удалены после проверки. Исходники и отчёты сохранены. Случайные generated files для запуска или чтения static demo не требуются.

## Fake uvicorn smoke без ключа

Настоящий uvicorn запускался с `SALVADOR_TRANSCRIBER_BACKEND=fake`, пустым `SALVADOR_OPENAI_API_KEY` и отдельной SQLite-базой `var/final-smoke.db`.

| Проверка | Фактический результат |
|---|---|
| `GET /health/ready` | HTTP 200, `{"status":"ok","backend":"fake"}` |
| `GET /demo` | HTTP 200; страница содержит независимые `guestFile` и `operatorFile` |
| `GET /metrics/state` | HTTP 200, JSON получен, активных звонков 0 |
| Остановка | В логе есть `Application shutdown complete` и `Finished server process`; shell зафиксировал ожидаемый status 143 после SIGTERM |

WebSocket lifecycle, partial/final event stream и сохранение финала дополнительно проверены в автоматическом наборе через TestClient. Браузерная передача реальных файлов и настоящая OpenAI-транскрибация в smoke не выполнялись.

## Live-проверки OpenAI

Дата реальных вызовов: 2026-11-06. Наличие API key подтверждено только безопасным статусом; ключ, содержимое `.env` и Authorization headers не читались в отчёты и не выводились. TCP-подключение к `api.openai.com:443` было доступно. Каждый запуск [`scripts/live_scenarios.py`](../scripts/live_scenarios.py) явно использовал `SALVADOR_TRANSCRIBER_BACKEND=openai` и `--confirm-live-openai`; fake не применялся. В новом default-normal запуске `SALVADOR_OPENAI_MODEL` был удалён из окружения процесса, поэтому использовался именно default из [`Settings`](../src/salvador_transcriber/config.py:9), а не model override.

### Модель, VAD и тестовое аудио

- Первый normal с тогдашним default `gpt-live-transcribe` завершился неуспешно. Provider вернул sanitized error: `type=invalid_request_error`, `code=invalid_value`, `message=Turn detection is not supported for this transcription model.` Получено 0 partial/final; за 30-секундное ожидание зафиксированы 35 `OpenAIRealtimeProviderError`, повторные reconnect и 35 gaps. Сырой отчёт: [`var/live-normal-initial.json`](../var/live-normal-initial.json). Отклонение относится к текущему server-VAD профилю и не доказывает неработоспособность модели в других конфигурациях.
- В исходной серии всех последующих запусков явно задавался fallback `SALVADOR_OPENAI_MODEL=gpt-4o-transcribe`; отдельный probe получил `session.updated`. На тот момент default в коде и конфигурации не изменялся.
- После смены default на `gpt-4o-transcribe` повторно выполнен только `normal` без model override. Runner завершился со статусом `completed`; sanitized report [`var/live-normal-default.json`](../var/live-normal-default.json) подтверждает модель в `configuration.model`. Остальные live-сценарии повторно не запускались.
- VAD: `server_vad`, threshold 0.5, prefix padding 300 мс, silence duration 700 мс. Входные файлы — mono WAV PCM16LE, 16000 Гц, 20-мс кадры в реальном темпе; сервис ресэмплировал их в 24000 Гц.
- Локальные espeak/espeak-ng/piper/festival/flite/RHVoice/pico2wave и ffmpeg flite отсутствовали. Поэтому подготовлены семь коротких фраз через OpenAI TTS `gpt-4o-mini-tts`, затем локально сконвертированы ffmpeg. Использовалась осмысленная русская речь о бронировании, а не тоны или шум. Манифест и WAV находятся только в игнорируемом [`var/live-audio/`](../var/live-audio/).
- Для pauses физически собраны 300 мс тишины внутри первой фразы и 1600 мс между фразами; итоговая длительность 14.620 с. Для reconnect после разрыва использована отдельная запись 6.464 с. Два звонка имеют различимые независимые тексты.
- Сырые JSON, SQLite и logs сохранены только в игнорируемом `var/`. Исходное сканирование 29 файлов в `var/` не нашло API key, `Authorization:` или `Bearer `-значений (0 совпадений). Новые default-normal JSON/log дополнительно проверены на маркеры API key, `Authorization` и `Bearer`: 0 совпадений. Файл `.env` не изменялся.

### Фактические сценарии

| Сценарий / отчёт | Статус | Фактический результат |
|---|---|---|
| REAL-S1 normal, прежний default | **Failed: provider/VAD configuration** | Ошибка `gpt-live-transcribe` в текущем server-VAD профиле описана выше; транскрипция и latency отсутствуют, сбой не засчитан как успех |
| REAL-S1 normal, исходный explicit fallback — [`JSON`](../var/live-normal-gpt4o.json) | **Pass на gpt-4o-transcribe** | 27 partial, 2 final, 0 errors/gaps. Guest: «Здравствуйте, я хочу забронировать номер на две ночи.» Operator: «Добрый день, подтверждаю бронирование стандартного номера.» Роли и смысл точные |
| REAL-S1 normal, новый default без model override — [`JSON`](../var/live-normal-default.json) | **Pass** | `gpt-4o-transcribe` из default config; 27 partial, 2 final, 0 scenario/fatal errors и 0 gaps. Тексты guest/operator совпали с исходным normal, роли и смысл точные |
| REAL-S2 overlap — [`JSON`](../var/live-overlap-gpt4o.json) | **Pass** | 27 partial, 2 final, 0 errors/gaps. Тексты совпадают с normal и не смешаны. Границы guest 0–5344 мс и operator 276–5568 мс действительно перекрываются; utterance IDs независимы |
| REAL-S3 pauses, первая fixture — [`JSON`](../var/live-pauses-gpt4o.json) | **Runner completed; VAD expectation failed** | Получены 3 финала: первая смысловая фраза разделилась. Краевые 200+200 мс вокруг вставленных 300 мс дали фактические ~700 мс, равные VAD silence duration. Все 3 latency-наблюдения сохранены в общей статистике |
| REAL-S3 pauses, исправленная fixture — [`JSON`](../var/live-pauses-300ms-gpt4o.json) | **Pass** | 30 partial, 2 final, 0 errors/gaps. Реальные 300 мс не разделили первую фразу: «Я хочу уточнить детали бронирования. Для двух гостей на две ночи.» Длинная пауза завершила её перед второй фразой о бронировании |
| REAL-S4 reconnect — [`JSON`](../var/live-reconnect-gpt4o.json) | **Pass** | 53 partial, 4 final. Demo-only endpoint закрыл именно provider WebSocket guest. Одна попытка и один успешный reconnect; старые финалы неизменны, новый guest final в generation 2: «После восстановления связи прошу подтвердить мою бронь.» Operator выдал второй final в generation 1 |
| REAL-S5 two_calls — [`JSON`](../var/live-two-calls-gpt4o.json) | **Pass** | 56 partial, 4 final, 2 `call_completed`, 0 errors/gaps. `live-two-a/hotel-live-two-a` и `live-two-b/hotel-live-two-b` содержат только свои различимые фразы; sequence каждого звонка независим. После закрытия active calls/channels = 0/0 |

Прямые запросы к SQLite подтвердили по одной строке на уникальный final: каждый normal 2, overlap 2, corrected pauses 2, reconnect 4, two_calls 4. Partial в `final_utterances` отсутствуют. В reconnect сохранена одна строка `audio_gaps`; в остальных успешных сценариях gaps нет. Повторных `call_completed` нет.

В успешных логах встречаются только предупреждения об игнорировании необязательных provider events `conversation.item.added` и `conversation.item.done`; ошибок закрытия, timeout и traceback в этих запусках нет. Эти events не требуются для partial/final и покрыты существующей generic unknown-event проверкой.

### Live latency

Использованы фактические наблюдения [`BoundedLatencyCollector`](../src/salvador_transcriber/observability/latency.py:58): от monotonic-времени ingress 20-мс кадра, содержащего подтверждённый VAD `end_ms`, до внутреннего финала после сохранения. Агрегаты вычислены [`summarize_latency()`](../src/salvador_transcriber/observability/latency.py:132), p95 — nearest-rank. Длительность файла и момент отправки всего аудио не подменяли измерение.

| Набор | count | min, мс | median, мс | p95, мс | max, мс | ≤2000 мс | unavailable |
|---|---:|---:|---:|---:|---:|---:|---:|
| Исходный успешный REAL-S1 normal, explicit fallback | 2 | 941.359 | 973.225 | 1005.090 | 1005.090 | 2/2 = 100% | 0 |
| Новый REAL-S1 normal, default без model override | 2 | 694.700 | 1056.361 | 1418.022 | 1418.022 | 2/2 = 100% | 0 |
| Все normal-connection запуски: normal + overlap + обе pauses + two_calls | 13 | 607.565 | 729.823 | 1551.652 | 1551.652 | 13/13 = 100% | 0 |
| Из них guest | 9 | 607.565 | 766.915 | 1551.652 | 1551.652 | 9/9 = 100% | 0 |
| Из них operator | 4 | 666.563 | 698.368 | 941.359 | 941.359 | 4/4 = 100% | 0 |
| REAL-S4 reconnect, все финалы до и после разрыва | 4 | 803.479 | 866.867 | 890.879 | 890.879 | 4/4 = 100% | 0 |
| Только post-reconnect guest, generation 2 | 1 | 890.879 | 890.879 | 890.879 | 890.879 | 1/1 = 100% | 0 |

Медленные валидные наблюдения не удалялись: первая pauses-попытка включена в общий count 13. В failed attempt прежнего default финалов не было, поэтому latency observations и unavailable records также нет; это не нулевая задержка. Два наблюдения нового default-normal показаны отдельно и не смешаны с прежними агрегатами. Выборка меньше целевых 20 финалов на роль из плана, поэтому подтверждён только фактический p95 этой ограниченной выборки, а не статистически надёжный production SLO.

### Reconnect timing и потери

- Counter deltas: disconnects 1, attempts 1, successes 1. Фактический `reconnect_duration_seconds`: count 1, min/sum/max 1.024337 с.
- Gap `openai_disconnect` относится к guest generation 1; `start_ms/end_ms=null`, dropped frames/bytes = 0/0. Аудио во время восстановления не подавалось, поэтому отсутствие dropped audio подтверждено, а неизвестные границы не выдумываются.
- Время от сохранённого `occurred_at` gap до первого нового guest final — 7.781577 с. Оно включает подачу 6.464-секундной записи в реальном темпе и не является чистым reconnect duration.
- Точные отдельные `t_disconnect`, `t_detected` и `t_ready` runner не записывает; detection delay и время готовности сверх histogram **unavailable**. Export timing также unavailable, поскольку exporter был отключён.

### Исправления и оставшиеся ограничения live

- В исходной live-серии была исправлена только тестовая pauses fixture: убраны краевые padding-и на внутреннем стыке. В текущем исправлении изменён только default на `gpt-4o-transcribe` в [`Settings`](../src/salvador_transcriber/config.py:9) и [`.env.example`](../.env.example); provider protocol и VAD не менялись. Добавлен focused config regression, полный набор теперь включает 84 проходящих теста.
- Отклонение `gpt-live-transcribe` провайдером задокументировано только для configured `server_vad`; это не утверждение о других профилях. Текущий default `gpt-4o-transcribe` проверен отдельным normal без `SALVADOR_OPENAI_MODEL`; явный override по-прежнему поддерживается.
- Проверены короткие синтетические TTS-записи, а не живые голоса/телефония. Нагрузка, длительная стабильность, реальные сетевые помехи и целевая выборка 20 финалов на роль не проверялись.
- Настоящий Salvador webhook не проверялся; его данные не предоставлены. HTTP export проверен только с локальным mock. Внешняя доставка остаётся at-least-once с idempotency key и без durable outbox после перезапуска.

## Git и игнорируемые данные

Git 2.56.0 доступен; выполнен только `git init`, глобальная конфигурация не изменялась и commit не создавался. Ветка `master` не имеет коммитов. Исходные каталоги и файлы проекта показаны как untracked; `.env`, `.venv/`, `*.egg-info/` и `var/` подтверждённо игнорируются через [`.gitignore`](../.gitignore), а `.env.example` остаётся доступным для отслеживания.
