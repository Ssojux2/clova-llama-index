import requests
import json
import time
import uuid
from typing import List, Dict, Any, Optional

class ClovaEmbeddings:
    """
    HyperCLOVA Embedding v2 엔드포인트와 상호작용하는 헬퍼 클래스입니다.
    ClovaClient 내부에서 사용됩니다.
    """
    def __init__(self, client: 'ClovaClient', request_delay: float = 0.75):
        """
        ClovaEmbeddings 헬퍼를 초기화합니다.

        Args:
            client: ClovaClient 인스턴스.
            request_delay: 각 요청 전에 대기할 시간(초). 기본값 0.75초.
                           (참고: 네이버 클라우드 API는 초당 요청 수 제한(TPS)이 있을 수 있습니다.)
        """
        self.client = client
        self.request_delay = request_delay
        # 임베딩 엔드포인트 기본 URL 명시적 저장
        self.embedding_url = f"{self.client.base_url}/testapp/v1/api-tools/embedding/v2/"

    def create(self, input: str, model: str = "embedding_v2") -> Dict[str, Any]:
        """
        HyperCLOVA Embedding v2 엔드포인트를 호출하고 응답을 반환합니다.

        Args:
            input: 임베딩할 텍스트.
            model: 임베딩 모델 이름 (현재 엔드포인트 로직에서는 사용되지 않으나, 향후 호환성을 위해 유지).

        Returns:
            OpenAI 응답 형식과 유사하게 포맷된 임베딩 및 사용량 정보를 포함하는 딕셔너리.

        Raises:
            ValueError: 서버가 빈 응답을 반환하거나, JSON 파싱에 실패하거나, API 오류 상태를 반환하는 경우.
            requests.exceptions.RequestException: API 호출 중 네트워크 관련 오류 발생 시.
        """
        # 요청 전 딜레이
        time.sleep(self.request_delay)

        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()) # 고유 요청 ID 생성
        }
        payload = {
            "text": input
        }

        try:
            response = requests.post(self.embedding_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # 4xx 또는 5xx 응답 시 HTTPError 발생
        except requests.exceptions.RequestException as e:
            print(f"네트워크 또는 HTTP 오류 발생: {e}")
            raise

        if not response.text:
            raise ValueError("서버로부터 빈 응답을 받았습니다.")

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            print("JSON 응답 파싱 실패. 응답 텍스트:", response.text)
            raise ValueError("JSON 응답 파싱에 실패했습니다.") from e

        # 응답 본문에 API 관련 오류가 있는지 확인
        if "status" in data and data["status"].get("code") != "20000":
            err_msg = data["status"].get("message", "알 수 없는 API 오류")
            raise ValueError(f"HyperCLOVA Embedding API 오류: {err_msg}")

        result = data.get("result", {})
        embedding = result.get("embedding", [])
        input_tokens = result.get("inputTokens", 0)

        # OpenAI Embedding API 응답과 유사한 형식으로 출력 포맷팅
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embedding
                }
            ],
            "model": model, # 요청된 모델 이름 반환
            "usage": {
                "prompt_tokens": input_tokens,
                "total_tokens": input_tokens # 임베딩 API는 보통 입력 토큰만 계산
            }
        }

class ClovaClient:
    """
    네이버 클라우드 HyperCLOVA X API (Chat Completion 및 Embeddings)와 상호작용하기 위한 클라이언트입니다.
    """
    def __init__(self, api_key: str, base_url: str = "https://clovastudio.stream.ntruss.com"):
        """
        ClovaClient를 초기화합니다.

        Args:
            api_key: HyperCLOVA X 용 네이버 클라우드 API 키.
            base_url: Clova Studio API의 기본 URL.
        """
        if not api_key:
            raise ValueError("API 키는 비어 있을 수 없습니다.")
        self.api_key = api_key
        self.base_url = base_url
        # 임베딩 헬퍼 클래스 인스턴스화
        self.embeddings = ClovaEmbeddings(self)

    def chat_completions_create(self, model: str, messages: List[Dict[str, str]], n: int = 1, **kwargs: Any) -> Dict[str, Any]:
        """
        HyperCLOVA X Chat Completion API를 호출합니다.

        Args:
            model: 사용할 모델 ID (예: "HCX-003").
            messages: OpenAI 형식의 메시지 객체 리스트
                      (예: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, ...]).
            n: 생성할 응답(completion)의 수. 기본값은 1입니다.
               참고: HyperCLOVA API는 단일 호출에서 n > 1을 지원하지 않을 수 있습니다.
               이 함수는 n > 1인 경우 여러 번 호출하여 이를 시뮬레이션합니다.
            **kwargs: API에 전달할 추가 파라미터:
                temperature (float): 샘플링 온도.
                maxTokens (int): 생성할 최대 토큰 수.
                topK (int): Top-K 샘플링 파라미터.
                topP (float): Top-P (nucleus) 샘플링 파라미터.
                repeatPenalty (float): 토큰 반복에 대한 페널티.
                includeAiFilters (bool): AI 필터 사용 여부.
                stopBefore (List[str]): 생성을 중단할 문자열 리스트.
                seed (int): 무작위 생성을 위한 시드값.

        Returns:
            API 응답을 포함하는 딕셔너리. n > 1인 경우, 'result.message'는 각 개별 API 호출의
            메시지 객체를 포함하는 리스트가 됩니다.

        Raises:
            ValueError: 서버가 빈 응답을 반환하거나 JSON 파싱에 실패하는 경우.
            requests.exceptions.RequestException: API 호출 중 네트워크 관련 오류 발생 시.
        """
        # API 엔드포인트 URL 구성
        url = f"{self.base_url}/testapp/v1/chat-completions/{model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
            # 참고: 필요시 X-NCP-CLOVASTUDIO-REQUEST-ID 헤더 추가 가능
        }

        # 기본 파라미터 (kwargs로 덮어쓸 수 있음)
        default_params = {
            "temperature": 0.8, # 일반적인 사용례에 기반한 기본값 조정
            "maxTokens": 512,   # 기본값 조정
            "topK": 0,          # 0이면 사용 안 함
            "topP": 0.8,        # 기본값 조정
            "repeatPenalty": 5.0, # 적절한 값으로 설정 (API 문서 확인 필요)
            "includeAiFilters": True,
            "stopBefore": [],
            "seed": 0           # 0이면 랜덤 시드 사용 가능성 있음 (API 문서 확인)
        }

        if n == 1:
            # 단일 요청
            payload = {
                "messages": messages,
                **default_params, # 기본값 먼저 적용
                **kwargs          # 사용자가 제공한 kwargs로 덮어쓰기
            }
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status() # 오류 응답 시 예외 발생
            except requests.exceptions.RequestException as e:
                print(f"Chat Completion 중 네트워크 또는 HTTP 오류 발생: {e}")
                raise

            if not response.text:
                raise ValueError("Chat Completion 서버로부터 빈 응답을 받았습니다.")
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                print("Chat Completion의 JSON 응답 파싱 실패. 응답 텍스트:", response.text)
                raise ValueError("Chat Completion JSON 응답 파싱에 실패했습니다.") from e

            # 일관성을 위해 result.message가 항상 리스트이도록 보장 (비어 있거나 단일 객체인 경우 포함)
            result = data.get("result", {})
            if "message" in result:
                if not isinstance(result["message"], list):
                    # API가 때때로 단일 dict를 반환하는 경우 리스트로 감싸기
                    result["message"] = [result["message"]]
            else:
                # API가 오류 또는 빈 응답 시 message 필드를 생략하는 경우에도 필드 존재 보장
                result["message"] = []

            data["result"] = result # 수정된 result를 다시 할당
            return data

        else:
            # n > 1 시뮬레이션: 여러 번의 요청 생성
            messages_list = []
            # 초기 시드값 가져오기 (kwargs에 없으면 기본값 사용)
            current_seed = kwargs.get("seed", default_params["seed"])

            for i in range(n):
                payload = {
                    "messages": messages,
                    **default_params,   # 기본값 먼저 적용
                    **kwargs,           # 사용자 kwargs 적용 (기본값 덮어쓰기 가능)
                    "seed": current_seed + i # 각 요청마다 시드값 증가 (API가 시드값을 다르게 처리해야 의미 있음)
                }
                try:
                    response = requests.post(url, headers=headers, data=json.dumps(payload))
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Chat Completion 중 네트워크 또는 HTTP 오류 발생 (n={i+1}): {e}")
                    # 즉시 예외를 발생시킬지, 부분 결과를 수집할지 결정
                    # 여기서는 오류 플레이스홀더를 추가하거나 건너뛰기
                    messages_list.append({"role": "assistant", "content": f"[오류: 요청 {i+1} 실패: {e}]"})
                    continue # 다음 요청으로 이동

                if not response.text:
                    print(f"Chat Completion 서버로부터 빈 응답 수신 (n={i+1}).")
                    messages_list.append({"role": "assistant", "content": f"[오류: 요청 {i+1} 빈 응답 반환]"})
                    continue

                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    print(f"Chat Completion JSON 응답 파싱 실패 (n={i+1}). 응답 텍스트:", response.text)
                    messages_list.append({"role": "assistant", "content": f"[오류: 요청 {i+1} JSON 파싱 실패]"})
                    continue

                result = data.get("result", {})
                # message 내용 추출, 없을 수 있거나 리스트가 아닌 경우 처리
                if "message" in result:
                    if isinstance(result["message"], list) and len(result["message"]) > 0:
                        messages_list.append(result["message"][0]) # 첫 번째 메시지 dict 추가
                    elif isinstance(result["message"], dict):
                        messages_list.append(result["message"]) # 단일 메시지 dict 추가
                    else:
                        messages_list.append({"role": "assistant", "content": "[오류: 예상치 못한 메시지 형식]"})
                else:
                    messages_list.append({"role": "assistant", "content": "[오류: 결과에서 메시지를 찾을 수 없음]"})

            # 결과를 단일 응답 구조로 결합
            # 참고: 사용량 통계(토큰)는 여기서 정확하게 집계되지 않음.
            # 이 형식은 생성된 메시지 반환에 중점을 둡니다.
            combined_data = {
                "result": {
                    "message": messages_list,
                    # 이 결과가 여러 호출에서 온 것임을 나타냄
                    "stopReason": "multiple_calls_simulated_n"
                },
                "status": {"code": "20000", "message": f"OK (Simulated n={n})"} # 성공 상태로 가정
            }
            return combined_data