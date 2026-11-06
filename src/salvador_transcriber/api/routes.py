import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, status
from fastapi.responses import HTMLResponse
from pydantic import TypeAdapter, ValidationError

from salvador_transcriber.api.schemas import (
    CallCreateRequest,
    CallResponse,
    GapResponse,
    HealthResponse,
    IngressEnvelope,
    IngressResponse,
    PingCommand,
    ServiceEventResponse,
    StartCommand,
    StopCommand,
    TranscriptResponse,
    UtteranceResponse,
)
from salvador_transcriber.audio.pcm import AudioValidationError
from salvador_transcriber.demo.events import DemoEventHub
from salvador_transcriber.demo.page import demo_html
from salvador_transcriber.domain.events import EventType
from salvador_transcriber.domain.state import Role
from salvador_transcriber.sessions.errors import (
    CallAlreadyExistsError,
    CallNotFoundError,
    SessionError,
    SessionStateError,
)
from salvador_transcriber.sessions.registry import CallRegistry
from salvador_transcriber.storage.repository import TranscriptNotFoundError, TranscriptRepository

router = APIRouter()
_start_adapter = TypeAdapter(StartCommand)


def _registry(request: Request) -> CallRegistry:
    registry: CallRegistry = request.app.state.registry
    return registry


def _repository(request: Request) -> TranscriptRepository:
    repository: TranscriptRepository = request.app.state.repository
    return repository


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo() -> HTMLResponse:
    return HTMLResponse(demo_html())


@router.get("/health/live", response_model=HealthResponse)
async def live(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", backend=request.app.state.settings.transcriber_backend)


@router.get("/health/ready", response_model=HealthResponse)
async def ready(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", backend=request.app.state.settings.transcriber_backend)


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics(request: Request) -> Response:
    if not request.app.state.settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return Response(
        content=request.app.state.metrics.prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/metrics/state", include_in_schema=False)
async def metrics_state(request: Request) -> dict[str, object]:
    if not request.app.state.settings.metrics_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    snapshot: dict[str, object] = request.app.state.metrics.snapshot()
    return snapshot


@router.post(
    "/calls",
    response_model=CallResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_call(request: Request, command: CallCreateRequest) -> CallResponse:
    try:
        call = await _registry(request).create_call(command.call_id, command.hotel_id)
    except CallAlreadyExistsError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    except SessionStateError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error
    return CallResponse(call_id=call.call_id, hotel_id=call.hotel_id, state=call.state)


@router.post("/calls/{call_id}/stop", response_model=CallResponse)
async def stop_call(request: Request, call_id: str) -> CallResponse:
    try:
        call = await _registry(request).close_call(call_id)
    except CallNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    return CallResponse(call_id=call.call_id, hotel_id=call.hotel_id, state=call.state)


@router.get("/calls/{call_id}/transcript", response_model=TranscriptResponse)
async def read_transcript(
    request: Request,
    call_id: str,
    hotel_id: Annotated[str, Query(min_length=1)],
) -> TranscriptResponse:
    try:
        transcript = await _repository(request).get_transcript(call_id, hotel_id)
    except TranscriptNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="call transcript not found",
        ) from error
    return TranscriptResponse(
        call_id=transcript.call_id,
        hotel_id=transcript.hotel_id,
        utterances=tuple(
            UtteranceResponse.model_validate(item, from_attributes=True)
            for item in transcript.utterances
        ),
        gaps=tuple(
            GapResponse.model_validate(item, from_attributes=True) for item in transcript.gaps
        ),
        service_events=tuple(
            ServiceEventResponse.model_validate(item, from_attributes=True)
            for item in transcript.service_events
        ),
    )


@router.websocket("/ws/calls/{call_id}/events")
async def call_events(
    websocket: WebSocket,
    call_id: str,
    hotel_id: Annotated[str, Query(min_length=1)],
) -> None:
    await websocket.accept()
    registry: CallRegistry = websocket.app.state.registry
    try:
        call = await registry.get_call(call_id)
        if call.hotel_id != hotel_id:
            raise CallNotFoundError(f"call {call_id} does not exist in this hotel")
    except CallNotFoundError:
        await _send_response(
            websocket,
            IngressResponse(type="error", code="call_not_found", message="call not found"),
        )
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    hub: DemoEventHub = websocket.app.state.demo_event_hub
    async with hub.subscribe(call_id, hotel_id) as queue:
        receive_task = asyncio.create_task(websocket.receive())
        try:
            while True:
                event_task = asyncio.create_task(queue.get())
                done, _ = await asyncio.wait(
                    (receive_task, event_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive_task in done:
                    event_task.cancel()
                    await asyncio.gather(event_task, return_exceptions=True)
                    break
                event = event_task.result()
                await websocket.send_json(event.model_dump(mode="json"))
                if event.event_type is EventType.CALL_COMPLETED:
                    await websocket.close()
                    break
        finally:
            receive_task.cancel()
            await asyncio.gather(receive_task, return_exceptions=True)


@router.post("/demo/calls/{call_id}/{role}/provider-disconnect", include_in_schema=False)
async def interrupt_provider_for_demo(
    request: Request,
    call_id: str,
    role: str,
    hotel_id: Annotated[str, Query(min_length=1)],
) -> dict[str, object]:
    settings = request.app.state.settings
    if not settings.demo_provider_faults_enabled or settings.transcriber_backend != "openai":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    try:
        channel_role = Role(role)
        call = await _registry(request).get_call(call_id)
        if call.hotel_id != hotel_id:
            raise CallNotFoundError(f"call {call_id} does not exist in this hotel")
        channel = call.channels[channel_role]
        generation = channel.connection_generation
        await channel.interrupt_provider_for_demo()
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid role",
        ) from error
    except CallNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
    except SessionError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
    return {"status": "accepted", "role": channel_role.value, "generation": generation}


async def _send_response(websocket: WebSocket, response: IngressResponse) -> None:
    await websocket.send_json(response.model_dump(mode="json"))


async def _reject(websocket: WebSocket, code: str, message: str) -> None:
    await _send_response(websocket, IngressResponse(type="error", code=code, message=message))
    await websocket.close(code=status.WS_1008_POLICY_VIOLATION)


@router.websocket("/ws/calls/{call_id}/{role}")
async def channel_ingress(websocket: WebSocket, call_id: str, role: str) -> None:
    await websocket.accept()
    try:
        channel_role = Role(role)
    except ValueError:
        await _reject(websocket, "invalid_role", "role must be guest or operator")
        return

    registry: CallRegistry = websocket.app.state.registry
    started = False
    try:
        first = await websocket.receive()
        if first["type"] == "websocket.disconnect":
            return
        if first.get("text") is None:
            await _reject(websocket, "start_required", "the first message must be start JSON")
            return
        try:
            start = _start_adapter.validate_json(first["text"])
            call = await registry.get_call(call_id)
            await call.start_channel(channel_role, start.hotel_id, start.audio)
            started = True
        except ValidationError as error:
            await _reject(websocket, "invalid_start", str(error))
            return
        except CallNotFoundError as error:
            await _reject(websocket, "call_not_found", str(error))
            return
        except SessionError as error:
            await _reject(websocket, "start_rejected", str(error))
            return

        await _send_response(
            websocket,
            IngressResponse(type="ready", code="ready", message=f"channel {channel_role} ready"),
        )
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                try:
                    await call.accept_audio(channel_role, start.hotel_id, message["bytes"])
                except AudioValidationError as error:
                    await _reject(websocket, error.code, str(error))
                    break
                except SessionError as error:
                    await _reject(websocket, "audio_rejected", str(error))
                    break
                continue
            if message.get("text") is None:
                await _reject(websocket, "invalid_message", "expected PCM bytes or command JSON")
                break
            try:
                command = IngressEnvelope.model_validate_json(message["text"]).root
                if isinstance(command, PingCommand):
                    await _send_response(
                        websocket,
                        IngressResponse(type="pong", code="pong", message="pong"),
                    )
                elif isinstance(command, StopCommand):
                    await registry.close_call(call_id)
                    await _send_response(
                        websocket,
                        IngressResponse(
                            type="accepted",
                            code="call_completed",
                            message="call completed",
                        ),
                    )
                    break
                else:
                    await _reject(websocket, "unexpected_start", "channel is already started")
                    break
            except ValidationError as error:
                await _reject(websocket, "invalid_command", str(error))
                break
            except SessionError as error:
                await _reject(websocket, "command_rejected", str(error))
                break
    finally:
        if started:
            await call.close_channel(channel_role)
