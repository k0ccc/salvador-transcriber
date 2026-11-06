# Salvador Transcriber

Небольшой асинхронный сервис двухканальной потоковой транскрибации телефонного разговора. Он принимает **раздельные** PCM-потоки `guest` и `operator`, создаёт независимые сессии транскрибации, публикует partial/final-события, сохраняет финальную транскрипцию и gaps в SQLite и может экспортировать финалы в Salvador AI.

Телефония, SIP, CRM, RAG и генерация подсказок не входят в scope. Реальный OpenAI Realtime API и сценарный fake — разные backend: fake предназначен для автотестов и проверки транспорта, а не для доказательства качества или задержки OpenAI.

## Требования

- Python 3.11 или новее.
- Linux/macOS или другое окружение с поддержкой Python WebSocket.
- Для реальной транскрибации — действующий OpenAI API key и сетевой доступ к OpenAI Realtime API.
- Входное аудио: mono PCM16 little-endian, 8000, 16000 или 48000 Гц. Сервис ресэмплирует каждый канал независимо в 24000 Гц для OpenAI.

## Установка в чистом окружении

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
cp .env.example .env
```

На Windows активируйте окружение и используйте соответствующий путь к Python вместо `.venv/bin/python`.

## Настройка

Все настройки сервиса имеют префикс `SALVADOR_`; пример без секретов находится в [`.env.example`](.env.example). Файл `.env` не коммитится.

### Fake без ключа

```dotenv
SALVADOR_TRANSCRIBER_BACKEND=fake
SALVADOR_OPENAI_API_KEY=
```

Fake принимает аудио и позволяет тестам сценарно выдавать события. Обычный fake в демонстрационной странице **не придумывает транскрипцию**.

### Настоящий OpenAI

```dotenv
SALVADOR_TRANSCRIBER_BACKEND=openai
SALVADOR_OPENAI_API_KEY=вставьте_свой_ключ_локально
SALVADOR_OPENAI_MODEL=gpt-4o-transcribe
SALVADOR_OPENAI_TRANSCRIPTION_LANGUAGE=ru
SALVADOR_OPENAI_VAD_THRESHOLD=0.5
SALVADOR_OPENAI_VAD_PREFIX_PADDING_MS=300
SALVADOR_OPENAI_VAD_SILENCE_DURATION_MS=700
```

Режим `openai` без ключа завершается ошибкой конфигурации и не переключается на fake. Live-runner также принимает стандартную переменную `OPENAI_API_KEY`; для uvicorn используется `SALVADOR_OPENAI_API_KEY`. Не передавайте ключ в URL, не записывайте его в отчёты и не коммитьте `.env`.

Безопасный default — `gpt-4o-transcribe`; `SALVADOR_OPENAI_MODEL` по-прежнему позволяет явно выбрать другую модель. В текущем профиле с `server_vad` провайдер отклонил `gpt-live-transcribe` с ошибкой «Turn detection is not supported for this transcription model», а `gpt-4o-transcribe` прошёл все пять live-сценариев. Это результат именно для текущего server-VAD профиля, а не утверждение, что `gpt-live-transcribe` не работает в других конфигурациях.

## Запуск

```bash
.venv/bin/python -m uvicorn salvador_transcriber.main:app \
  --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

OpenAPI доступен по адресу `http://127.0.0.1:8000/docs`, локальная демонстрация — по адресу `http://127.0.0.1:8000/demo`.

## Минимальный двухканальный протокол

1. Создайте звонок через `POST /calls` с JSON `{"call_id":"call-1","hotel_id":"hotel-1"}`.
2. Откройте **два** WebSocket: `/ws/calls/call-1/guest` и `/ws/calls/call-1/operator`.
3. Первым текстовым сообщением каждого соединения отправьте `start`.
4. Передавайте бинарные PCM16LE-кадры соответствующего источника. Не объединяйте дорожки.
5. Для завершения отправьте `{"type":"stop"}` в одном канале; сервер завершит оба канала звонка.

Пример с независимой передачей двух уже подготовленных PCM-файлов:

```python
import asyncio
import json

import httpx
from websockets.asyncio.client import connect

RATE = 16000
FRAME_BYTES = RATE * 20 // 1000 * 2


async def start(websocket, hotel_id):
    await websocket.send(
        json.dumps(
            {
                "type": "start",
                "hotel_id": hotel_id,
                "audio": {
                    "encoding": "pcm16le",
                    "sample_rate": RATE,
                    "channels": 1,
                    "frame_duration_ms": 20,
                },
            }
        )
    )
    assert json.loads(await websocket.recv())["type"] == "ready"


async def send_pcm(websocket, path):
    pcm = open(path, "rb").read()
    started = asyncio.get_running_loop().time()
    for index, offset in enumerate(range(0, len(pcm), FRAME_BYTES), 1):
        await websocket.send(pcm[offset : offset + FRAME_BYTES])
        delay = started + index * 0.02 - asyncio.get_running_loop().time()
        if delay > 0:
            await asyncio.sleep(delay)


async def main():
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8000") as client:
        response = await client.post("/calls", json={"call_id": "call-1", "hotel_id": "hotel-1"})
        response.raise_for_status()
    async with (
        connect("ws://127.0.0.1:8000/ws/calls/call-1/guest") as guest,
        connect("ws://127.0.0.1:8000/ws/calls/call-1/operator") as operator,
    ):
        await start(guest, "hotel-1")
        await start(operator, "hotel-1")
        await asyncio.gather(
            send_pcm(guest, "guest.pcm"),
            send_pcm(operator, "operator.pcm"),
        )
        await guest.send(json.dumps({"type": "stop"}))
        print(await guest.recv())


asyncio.run(main())
```

Ответы ingress: `ready`, `pong`, `accepted` и `error`. Нормализованные события не смешиваются с подтверждениями приёма: для локального просмотра откройте `/ws/calls/call-1/events?hotel_id=hotel-1`. События `transcript_partial`, `transcript_final`, `audio_gap` и `call_completed` содержат контекст звонка; partial/final дополнительно содержат `utterance_id`, `role`, `sequence`, `revision`, `start_ms`, `end_ms` и `connection_generation`.

## Локальная демонстрация

1. Запустите сервер в режиме `openai` для настоящего текста или в режиме `fake` для smoke-проверки транспорта.
2. Откройте `http://127.0.0.1:8000/demo`.
3. Выберите независимые mono WAV или raw `.pcm` для guest и operator. Для raw PCM укажите частоту каждого файла.
4. Нажмите «Запустить звонок». Страница передаёт 20-мс кадры в реальном темпе по двум WebSocket и показывает partial, final и gaps.
5. Остановите звонок кнопкой либо дождитесь окончания файлов, затем нажмите «Прочитать финальную транскрипцию».

Stereo WAV отклоняется: браузер не выполняет downmix и не смешивает участников. Внешний frontend-фреймворк и CDN не используются.

## Финальная транскрипция

```bash
curl --get 'http://127.0.0.1:8000/calls/call-1/transcript' \
  --data-urlencode 'hotel_id=hotel-1'
```

Ответ содержит:

- `utterances` — только финальные реплики, отсортированные по `start_ms`, `sequence`, `utterance_id`;
- `gaps` — причины, известные границы и число отброшенных кадров/байт;
- `service_events` — например, `call_completed`;
- `export_status` и `export_attempts` для каждой финальной реплики.

Partial не сохраняется как часть финальной транскрипции. Повторный финал с тем же `call_id` и `utterance_id` не создаёт новую запись.

## Метрики

- `GET /metrics` — Prometheus text format;
- `GET /metrics/state` — JSON-снимок счётчиков, гистограмм, активных звонков, состояний и глубины очередей.

Метрики включают принятые/отброшенные кадры, overflow, partial/final, дубли, ошибки, reconnect, gaps, обработку кадра и экспорт. Идентификаторы звонков, гостиниц и реплик не используются как Prometheus labels.

`final_latency_seconds` измеряется только когда провайдер подтвердил конец речи через VAD и этот `end_ms` удалось связать с monotonic-временем приёма соответствующего аудио ingress. Иначе увеличивается `final_latency_unavailable_total`. Длительность файла и момент отправки всего аудио не подменяют задержку. Фактические live-наблюдения 2026-11-06 и ограничения их небольшой выборки приведены в [`docs/test-results.md`](docs/test-results.md); они не заменяют production SLO.

## Salvador export

```dotenv
SALVADOR_EXPORTER_ENABLED=true
SALVADOR_EXPORTER_ENDPOINT=http://127.0.0.1:9000/events
SALVADOR_EXPORTER_TIMEOUT_SECONDS=5
SALVADOR_EXPORTER_MAX_ATTEMPTS=3
SALVADOR_EXPORTER_QUEUE_LIMIT=1000
```

Guest преобразуется в `guest_utterance_completed`, operator — в `operator_utterance_completed`. HTTP-запрос содержит контекст, текст и аудиограницы, а заголовок `Idempotency-Key` стабилен при retry. Сначала финал сохраняется в SQLite, затем ставится в ограниченную очередь экспорта. Семантика внешней доставки — at-least-once с конечным числом попыток; принимающая сторона должна дедуплицировать по ключу. Ошибка экспорта не удаляет локальный финал.

## Автоматические проверки

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy
.venv/bin/python -m ruff format --check .
```

Тесты не требуют сети и OpenAI key. Фактические результаты и отдельный live-статус зафиксированы в [`docs/test-results.md`](docs/test-results.md). Соответствие каждого обязательного требования реализации и проверкам приведено в [`docs/requirements-coverage.md`](docs/requirements-coverage.md).

## Реальные сценарии и сырой отчёт

Runner сам поднимает настоящий uvicorn backend с `openai`, подаёт файлы в реальном темпе и записывает JSON с событиями, транскрипциями, метриками, latency, reconnect, gaps и ошибками. Он не импортируется обычными тестами как live-проверка и не запускается без явных файлов, ключа и подтверждения.

В реальном прогоне 2026-11-06 прежний default `gpt-live-transcribe` отклонил configured `server_vad` (`invalid_value`: turn detection is not supported), а `gpt-4o-transcribe` прошёл все пять live-сценариев. Поэтому `gpt-4o-transcribe` теперь является default, а проверенная команда ниже не задаёт model override. После изменения default отдельно повторён только `normal`: он успешно завершился без model override. При явном выборе другой модели сначала проверьте поддержку текущего VAD-профиля и зафиксируйте её в отчёте.

```bash
export OPENAI_API_KEY='ваш_ключ_только_в_локальном_окружении'
.venv/bin/python scripts/live_scenarios.py \
  --scenario all \
  --guest-audio /path/guest.wav \
  --operator-audio /path/operator.wav \
  --second-guest-audio /path/second-guest.wav \
  --second-operator-audio /path/second-operator.wav \
  --after-disconnect-audio /path/after-disconnect.wav \
  --output var/live-report.json \
  --confirm-live-openai
```

Для одного сценария выберите `normal`, `overlap`, `pauses`, `reconnect` или `two_calls`; дополнительные файлы обязательны только для соответствующих сценариев. Для `pauses` паузы должны физически присутствовать в записи. Runner не оценивает смысл транскрипции за оператора: текст и проверки качества остаются в отчёте для ручного анализа. Отчёт может содержать текст разговора и пути к файлам — храните его как чувствительные данные.

Для `reconnect` runner включает demo-only endpoint, который закрывает именно provider WebSocket guest. В обычном сервере механизм выключен настройкой `SALVADOR_DEMO_PROVIDER_FAULTS_ENABLED=false`; не включайте его на незащищённом production-окружении.

Latency-отчёт содержит count/min/median/p95/max и долю до 2000 мс только для валидных наблюдений. При отсутствии подтверждённой VAD-границы статус — `unavailable`. Reconnect-наблюдения не смешиваются с normal-статистикой. Подробности — в [`docs/architecture.md`](docs/architecture.md).

## Текущие ограничения

- Пять live-сценариев выполнены 2026-11-06 с явным `gpt-4o-transcribe`; после смены default `normal` дополнительно прошёл без model override. Прежний default `gpt-live-transcribe` был отклонён провайдером в текущем `server_vad`-профиле. Это не доказывает неработоспособность модели вне данного профиля. Проверены короткие синтетические TTS-записи и ограниченная latency-выборка, а не живые голоса, телефония или production SLO. Подробности и failed attempt прежнего default — в [`docs/test-results.md`](docs/test-results.md).
- Fake-сценарии доказывают только детерминированную оркестрацию, хранение и обработку отказов.
- Реестр звонков и подписчики событий находятся в одном процессе. Распределённый реестр, sticky routing и горизонтальное масштабирование не реализованы.
- SQLite и in-process очереди не образуют durable outbox с повторной отправкой после перезапуска. Незавершённый экспорт при аварийном завершении требует отдельного recovery.
- Нагрузка комплекса на 4000 номеров не тестировалась; capacity planning требует пикового числа одновременных звонков и профиля аудио.
- Нет production authentication/authorization, TLS-терминации, HA и deployment-runbook.
- Demo рассчитано на локальную работу и mono WAV/PCM; это не браузерный кабинет и не телефонный адаптер.
- Точные поля OpenAI изолированы в адаптере и сверены с указанными в нём исходниками SDK, но это не заменяет реальный контрактный прогон с ключом.
