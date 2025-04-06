import asyncio # asyncio 임포트 추가
from typing import Any, List, Dict, Sequence, Optional
from pydantic import PrivateAttr, Field

from llama_index.core.llms.llm import LLM
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.callbacks import CallbackManager # 나중에 필요시 사용
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback # 특정 콜백

# 패키지 내에서 상대 경로로 ClovaClient 임포트
from .client import ClovaClient

# 모델별 기본값 설정 (HCX-003 기준, 필요시 API 문서 참고하여 조정)
DEFAULT_CONTEXT_WINDOW = 4000 # 예시 기본값, HCX-003의 정확한 값 확인 필요
DEFAULT_NUM_OUTPUT = 512    # 예시 기본값, API 기본값이나 사용자 선호에 따라 조정
DEFAULT_MODEL_NAME = "HCX-003" # 기본 모델

class ClovaLLM(LLM):
    """
    네이버 클라우드 HyperCLOVA X Chat Completion API를 사용하는 LlamaIndex LLM 클래스입니다.
    """
    model: str = Field(default=DEFAULT_MODEL_NAME, description="사용할 HyperCLOVA X 모델 이름")
    context_window: int = Field(default=DEFAULT_CONTEXT_WINDOW, description="모델의 최대 컨텍스트 창 크기")
    num_output: int = Field(default=DEFAULT_NUM_OUTPUT, description="기본 최대 생성 토큰 수")

    _client: ClovaClient = PrivateAttr()
    _api_kwargs: Dict[str, Any] = PrivateAttr() # 기본 API 호출 파라미터 저장을 위한 속성

    def __init__(
        self,
        client: ClovaClient,
        model: str = DEFAULT_MODEL_NAME,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        num_output: int = DEFAULT_NUM_OUTPUT,
        # 기본 API 호출에 사용할 추가 kwargs (예: temperature, topP 등)
        **kwargs: Any,
    ):
        """
        ClovaLLM 어댑터를 초기화합니다.
        (이하 __init__ 메서드 설명 생략 - 이전과 동일)
        """
        self._client = client
        self._api_kwargs = kwargs # 기본 API 파라미터 저장
        super().__init__(
            model=model,
            context_window=context_window,
            num_output=num_output,
        )

    @classmethod
    def class_name(cls) -> str:
        """클래스 이름을 반환합니다."""
        return "ClovaLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM 메타데이터를 반환합니다."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model,
            is_chat_model=True # Chat 엔드포인트를 지원함을 명시
        )

    # --- 내부 헬퍼 ---
    def _prepare_chat_messages(self, messages: Sequence[ChatMessage]) -> List[Dict[str, str]]:
        """LlamaIndex ChatMessage 시퀀스를 HyperCLOVA API 형식으로 변환합니다."""
        # (이하 _prepare_chat_messages 메서드 내용 생략 - 이전과 동일)
        role_map = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
        }
        api_messages = []
        for msg in messages:
            role = role_map.get(msg.role)
            if not role:
                print(f"경고: 처리할 수 없는 역할({msg.role})의 메시지를 건너<0xEB><0x9B><0x81>니다.")
                continue
            content = str(msg.content) if msg.content is not None else ""
            api_messages.append({"role": role, "content": content})
        if api_messages and not any(m['role'] != 'system' for m in api_messages):
             print("경고: Chat 메시지에 시스템 메시지만 포함되어 있습니다.")
        return api_messages


    # --- Chat 엔드포인트 (동기) ---
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """HyperCLOVA X API를 사용하여 채팅 대화를 시작합니다."""
        # (이하 chat 메서드 내용 생략 - 이전과 동일)
        api_messages = self._prepare_chat_messages(messages)
        if not api_messages:
             return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="[오류: 전송할 유효한 메시지가 없습니다]"))
        api_call_kwargs = {"maxTokens": self.num_output, **self._api_kwargs, **kwargs}
        try:
            response_data = self._client.chat_completions_create(
                model=self.model, messages=api_messages, n=1, **api_call_kwargs
            )
            result = response_data.get("result", {})
            message_list = result.get("message", [])
            if not message_list:
                status_info = response_data.get('status', '상태 정보 없음')
                error_content = f"[오류: Clova API가 메시지를 반환하지 않았습니다. 상태: {status_info}]"
                return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=error_content))
            assistant_message = message_list[0]
            content = assistant_message.get("content", "[오류: 응답 메시지에 내용이 없습니다]")
            final_message = ChatMessage(role=MessageRole.ASSISTANT, content=content)
            raw_response = response_data
            return ChatResponse(message=final_message, raw=raw_response)
        except Exception as e:
            print(f"Clova chat completion 중 오류 발생: {e}")
            return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=f"[채팅 생성 오류]: {e}"))

    # --- Chat 엔드포인트 (스트리밍, 동기) ---
    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """HyperCLOVA X API를 사용하여 채팅 대화를 스트리밍합니다 (시뮬레이션)."""
        # (이하 stream_chat 메서드 내용 생략 - 이전과 동일, 시뮬레이션 방식)
        print("경고: ClovaLLM은 현재 스트리밍을 시뮬레이션합니다...")
        api_messages = self._prepare_chat_messages(messages)
        if not api_messages:
            def empty_gen() -> ChatResponseGen:
                 delta = "[오류: 전송할 유효한 메시지가 없습니다]"
                 yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=delta), delta=delta)
                 return
            return empty_gen()
        api_call_kwargs = {"maxTokens": self.num_output, **self._api_kwargs, **kwargs}
        try:
            full_chat_response = self.chat(messages, **api_call_kwargs)
            full_text = full_chat_response.message.content or ""
            raw_response = full_chat_response.raw
            def gen() -> ChatResponseGen:
                response_so_far = ""
                if not full_text:
                    yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=""), delta="", raw=raw_response)
                    return
                for token in full_text:
                    response_so_far += token
                    is_last_chunk = (len(response_so_far) == len(full_text))
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=response_so_far),
                        delta=token,
                        raw=raw_response if is_last_chunk else None
                    )
            return gen()
        except Exception as e:
            print(f"시뮬레이션된 Clova 스트림 채팅 중 오류 발생: {e}")
            def error_gen() -> ChatResponseGen:
                error_msg = f"[채팅 생성 오류]: {e}"
                yield ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=error_msg), delta=error_msg)
                return
            return error_gen()

    # --- Completion 엔드포인트 (동기) ---
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """프롬프트를 사용자 메시지로 포맷하여 텍스트 완성을 수행합니다."""
        # (이하 complete 메서드 내용 생략 - 이전과 동일)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        chat_response = self.chat(messages, **api_call_kwargs)
        text = chat_response.message.content or ""
        return CompletionResponse(text=text, raw=chat_response.raw)

    # --- Completion 엔드포인트 (스트리밍, 동기) ---
    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """텍스트 완성을 스트리밍합니다 (시뮬레이션)."""
        # (이하 stream_complete 메서드 내용 생략 - 이전과 동일, 시뮬레이션 방식)
        print("경고: ClovaLLM은 현재 완성을 위해 스트리밍을 시뮬레이션합니다...")
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        stream_chat_response_gen = self.stream_chat(messages, **api_call_kwargs)
        def gen() -> CompletionResponseGen:
            full_response_text = ""
            final_raw = None
            for chat_chunk in stream_chat_response_gen:
                delta = chat_chunk.delta or ""
                full_response_text += delta
                if chat_chunk.raw is not None: final_raw = chat_chunk.raw
                yield CompletionResponse(
                    text=full_response_text,
                    delta=delta,
                    raw=final_raw # 마지막에만 전달될 것임
                )
        return gen()

    # --- 비동기 메서드 구현 ---

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """HyperCLOVA X API를 사용하여 비동기적으로 채팅 대화를 시작합니다."""
        # 동기 chat 메서드를 별도 스레드에서 실행
        return await asyncio.to_thread(self.chat, messages, **kwargs)

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """HyperCLOVA X API를 사용하여 비동기적으로 채팅 대화를 스트리밍합니다 (시뮬레이션)."""
        print("경고: ClovaLLM 비동기 스트리밍은 시뮬레이션됩니다.")
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        # 동기 chat 메서드를 별도 스레드에서 실행하여 전체 응답 받기
        full_chat_response = await asyncio.to_thread(self.chat, messages, **api_call_kwargs)
        full_text = full_chat_response.message.content or ""
        raw_response = full_chat_response.raw

        # 비동기 제너레이터 함수 정의
        async def gen() -> ChatResponseGen:
            response_so_far = ""
            if not full_text: # 응답 내용이 없는 경우
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                    delta="",
                    raw=raw_response
                )
                return

            # 문자(토큰 아님) 단위로 비동기적으로 yield
            for token in full_text:
                response_so_far += token
                is_last_chunk = (len(response_so_far) == len(full_text))
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=response_so_far),
                    delta=token,
                    raw=raw_response if is_last_chunk else None
                )
                await asyncio.sleep(0) # 다른 비동기 작업에 제어권 양보 (선택 사항)

        return gen()

    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """프롬프트를 사용자 메시지로 포맷하여 비동기적으로 텍스트 완성을 수행합니다."""
        # 동기 complete 메서드를 별도 스레드에서 실행
        return await asyncio.to_thread(self.complete, prompt, formatted=formatted, **kwargs)

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """비동기적으로 텍스트 완성을 스트리밍합니다 (시뮬레이션)."""
        print("경고: ClovaLLM 비동기 완성 스트리밍은 시뮬레이션됩니다.")
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        # 동기 complete 메서드를 별도 스레드에서 실행하여 전체 응답 받기
        full_completion_response = await asyncio.to_thread(self.complete, prompt, formatted=formatted, **api_call_kwargs)
        full_text = full_completion_response.text or ""
        raw_response = full_completion_response.raw

        # 비동기 제너레이터 함수 정의
        async def gen() -> CompletionResponseGen:
            response_so_far = ""
            if not full_text: # 응답 내용이 없는 경우
                yield CompletionResponse(text="", delta="", raw=raw_response)
                return

            # 문자(토큰 아님) 단위로 비동기적으로 yield
            for token in full_text:
                response_so_far += token
                is_last_chunk = len(response_so_far) == len(full_text)
                yield CompletionResponse(
                    text=response_so_far,
                    delta=token,
                    raw=raw_response if is_last_chunk else None
                )
                await asyncio.sleep(0) # 다른 비동기 작업에 제어권 양보 (선택 사항)

        return gen()