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

        Args:
            client: 초기화된 ClovaClient 인스턴스.
            model: HyperCLOVA X 모델 이름 (예: "HCX-003").
            context_window: 모델의 최대 컨텍스트 창 크기.
            num_output: 생성할 기본 최대 토큰 수.
            **kwargs: LLM 기본 클래스로 전달되거나 기본 API 호출 파라미터로 저장될 추가 인수.
                      (주의: LLM 기본 클래스의 파라미터와 충돌하지 않도록 주의)
        """
        self._client = client
        self._api_kwargs = kwargs # 기본 API 파라미터 저장

        # LLM 기본 클래스 초기화 시 필드 값 전달
        super().__init__(
            model=model,
            context_window=context_window,
            num_output=num_output,
            # callback_manager 등 LLM 기본 클래스가 받는 다른 kwargs 전달 가능
            # **kwargs #kwargs를 직접 전달하면 LLM 기본 클래스와 충돌 가능성 있음
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
        # LlamaIndex 역할(SYSTEM, USER, ASSISTANT)을 Clova 역할(system, user, assistant)로 변환
        # Clova가 OpenAI 표준과 유사한 소문자 역할을 사용한다고 가정
        role_map = {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            # FUNCTION, TOOL 등 Clova가 추후 지원 시 매핑 추가
        }
        api_messages = []
        for msg in messages:
            role = role_map.get(msg.role)
            if not role:
                print(f"경고: 처리할 수 없는 역할({msg.role})의 메시지를 건너<0xEB><0x9B><0x81>니다.")
                continue

            # content가 None이 아닌지 확인하고 문자열로 변환
            content = str(msg.content) if msg.content is not None else ""

            # 추가 kwargs 처리 (예: 함수 호출) - 필요시 구현
            # additional_kwargs = msg.additional_kwargs

            api_messages.append({"role": role, "content": content})

        # 기본 유효성 검사: API가 요구하는 경우 시스템 메시지 외 메시지가 최소 하나 있는지 확인
        if api_messages and not any(m['role'] != 'system' for m in api_messages):
             print("경고: Chat 메시지에 시스템 메시지만 포함되어 있습니다.")
             # API 동작에 따라 더미 사용자 메시지를 추가하거나 오류 발생시킬 수 있음

        return api_messages

    # --- Chat 엔드포인트 ---
    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """HyperCLOVA X API를 사용하여 채팅 대화를 시작합니다."""
        api_messages = self._prepare_chat_messages(messages)
        if not api_messages:
             # 모든 메시지가 건너뛰어졌거나 입력이 비어있는 경우 처리
             return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content="[오류: 전송할 유효한 메시지가 없습니다]"))

        # 기본/초기화 kwargs와 런타임 kwargs 결합 (런타임 우선)
        # self._api_kwargs는 __init__에서 받은 기본값들
        # kwargs는 chat() 호출 시 받은 특정 값들
        api_call_kwargs = {
            "maxTokens": self.num_output, # 기본 num_output 사용
            **self._api_kwargs,           # 초기화 시 설정된 기본값 적용
            **kwargs                      # 호출 시 전달된 값으로 덮어쓰기
        }

        try:
            response_data = self._client.chat_completions_create(
                model=self.model,
                messages=api_messages,
                n=1, # chat은 보통 하나의 응답을 기대
                **api_call_kwargs # 결합된 kwargs 전달
            )

            # 응답 처리
            result = response_data.get("result", {})
            message_list = result.get("message", [])

            if not message_list:
                # API 오류 또는 빈 응답 처리
                status_info = response_data.get('status', '상태 정보 없음')
                error_content = f"[오류: Clova API가 메시지를 반환하지 않았습니다. 상태: {status_info}]"
                return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=error_content))

            # n=1 이므로 리스트의 첫 번째 메시지가 주 응답이라고 가정
            assistant_message = message_list[0]
            content = assistant_message.get("content", "[오류: 응답 메시지에 내용이 없습니다]")

            # LlamaIndex ChatMessage 생성
            final_message = ChatMessage(
                role=MessageRole.ASSISTANT, # 항상 assistant 역할
                content=content,
                # additional_kwargs=assistant_message.get("additional_kwargs", {}) # Clova가 추가 데이터 반환 시
            )
            # 원시 응답 및 토큰 사용량 추출 (가능한 경우)
            raw_response = response_data
            # 토큰 사용량 정보가 응답에 포함되는 방식 확인 필요
            # usage_info = response_data.get("result", {}).get("inputLength", 0) # 예시: API 응답 형식 확인

            return ChatResponse(message=final_message, raw=raw_response) # usage 정보도 추가 가능

        except Exception as e:
            print(f"Clova chat completion 중 오류 발생: {e}")
            # ChatResponse 구조 내에서 오류 메시지 반환
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, content=f"[채팅 생성 오류]: {e}")
            )

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """HyperCLOVA X API를 사용하여 채팅 대화를 스트리밍합니다."""
        # 참고: 제공된 ClovaClient 코드는 스트리밍을 지원하지 않습니다.
        # 이 구현은 전체 채팅 메서드를 호출하고 결과를 토큰별로 생성하여 스트리밍을 *시뮬레이션*합니다.
        # 이것은 진정한 스트리밍이 아닙니다.
        # 실제 스트리밍을 위해서는 API가 지원하는 경우 (예: SSE - Server-Sent Events)
        # ClovaClient를 수정해야 합니다.

        print("경고: ClovaLLM은 현재 스트리밍을 시뮬레이션합니다. 실제 API 스트리밍은 클라이언트에 구현되지 않았습니다.")

        api_messages = self._prepare_chat_messages(messages)
        if not api_messages:
            def empty_gen() -> ChatResponseGen:
                 delta = "[오류: 전송할 유효한 메시지가 없습니다]"
                 yield ChatResponse(
                     message=ChatMessage(role=MessageRole.ASSISTANT, content=delta),
                     delta=delta
                 )
                 return
            return empty_gen()

        # 결합된 kwargs 준비 (chat 메서드와 동일)
        api_call_kwargs = {
            "maxTokens": self.num_output,
            **self._api_kwargs,
            **kwargs
        }

        try:
            # 전체 응답을 얻기 위해 비-스트리밍 메서드 호출
            # stream=True 같은 파라미터가 ClovaClient에 없으므로 chat() 사용
            full_chat_response = self.chat(messages, **api_call_kwargs)
            full_text = full_chat_response.message.content or ""
            raw_response = full_chat_response.raw # 전체 호출의 원시 응답 캡처

            # 청크를 생성하는 제너레이터 함수
            def gen() -> ChatResponseGen:
                response_so_far = ""
                if not full_text: # 응답 내용이 없는 경우 처리
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                        delta="",
                        raw=raw_response # 마지막에 raw 데이터 전달
                    )
                    return

                for token in full_text: # 문자별 반복 (실제 토큰 스트리밍 아님)
                    response_so_far += token
                    is_last_chunk = (len(response_so_far) == len(full_text))
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=response_so_far),
                        delta=token,
                        # 마지막 청크에서만 raw 데이터 전달
                        raw=raw_response if is_last_chunk else None
                    )
            return gen()

        except Exception as e:
            print(f"시뮬레이션된 Clova 스트림 채팅 중 오류 발생: {e}")
            def error_gen() -> ChatResponseGen:
                error_msg = f"[채팅 생성 오류]: {e}"
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=error_msg),
                    delta=error_msg
                )
                return
            return error_gen()


    # --- Completion 엔드포인트 (Chat을 Completion에 맞게 조정) ---
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        """프롬프트를 사용자 메시지로 포맷하여 텍스트 완성을 수행합니다."""
        # 프롬프트를 단일 사용자 메시지로 포맷하여 chat 엔드포인트 사용
        # 재정의되지 않는 한 기본 시스템 프롬프트 가정
        messages = [
            # 시스템 프롬프트 설정 가능하게 만들기 고려
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        # chat 메서드 호출 시 kwargs 전달
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        chat_response = self.chat(messages, **api_call_kwargs)
        text = chat_response.message.content or "" # 텍스트 내용 추출

        return CompletionResponse(
            text=text,
            raw=chat_response.raw # chat의 원시 응답 전달
            # 원시 데이터에 사용 가능한 경우 토큰 사용량 등 추가
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        """텍스트 완성을 스트리밍합니다."""
        # 비-스트리밍 complete 메서드를 호출하고 토큰별로 생성하여 스트리밍 시뮬레이션.
        # stream_chat의 주의사항 참조.

        print("경고: ClovaLLM은 현재 완성을 위해 스트리밍을 시뮬레이션합니다. 실제 API 스트리밍은 구현되지 않았습니다.")

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        # 내부적으로 스트리밍 채팅 메서드 사용
        # kwargs 전달
        api_call_kwargs = {**self._api_kwargs, **kwargs}
        stream_chat_response_gen = self.stream_chat(messages, **api_call_kwargs)

        # ChatResponseGen을 CompletionResponseGen으로 조정
        def gen() -> CompletionResponseGen:
            full_response_text = ""
            final_raw = None
            for chat_chunk in stream_chat_response_gen:
                delta = chat_chunk.delta or ""
                full_response_text += delta
                # 마지막 (잠재적으로 완전한) 원시 응답 추적
                # stream_chat의 gen()에서 마지막 청크에만 raw가 포함되도록 수정했으므로,
                # chat_chunk.raw가 None이 아닐 때가 마지막 청크임
                if chat_chunk.raw is not None:
                    final_raw = chat_chunk.raw

                yield CompletionResponse(
                    text=full_response_text,
                    delta=delta,
                    raw=final_raw if final_raw else None # 마지막 청크에서만 raw 전달
                )

        return gen()