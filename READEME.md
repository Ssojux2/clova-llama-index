# 네이버 클라우드 HyperCLOVA X 와 LlamaIndex 연동 (clova-llama-index)

이 패키지는 네이버 클라우드의 HyperCLOVA X 거대 언어 모델(LLM) 및 임베딩 API를 [LlamaIndex](https://github.com/run-llama/llama_index) 프레임워크와 통합하기 위한 헬퍼 클래스를 제공합니다.

**주요 기능:**

* `ClovaClient`: HyperCLOVA X Chat Completion 및 Embedding API에 대한 인증 및 요청을 처리하는 기본 클라이언트입니다.
* `ClovaIndexEmbeddings`: HyperCLOVA X Embedding API (v2)를 사용하는 LlamaIndex `BaseEmbedding` 호환 클래스입니다.
* `ClovaLLM`: HyperCLOVA X Chat Completion API (예: HCX-003)를 사용하는 LlamaIndex `LLM` 호환 클래스입니다.

**주의:** `ClovaLLM`의 스트리밍 메서드 (`stream_chat`, `stream_complete`)는 현재 전체 요청을 수행하고 결과를 토큰별로 반환하여 스트리밍을 *시뮬레이션*합니다. 진정한 API 스트리밍은 네이버 클라우드 API 지원 및 클라이언트 구현 업데이트에 따라 달라집니다.

## 설치

pip를 사용하여 GitHub에서 직접 이 패키지를 설치할 수 있습니다:

```bash
pip install git+[https://github.com/your-username/clova-llama-index.git](https://www.google.com/search?q=https://github.com/your-username/clova-llama-index.git)
```


```python
import os
# from dotenv import load_dotenv # 선택 사항: .env 파일에서 API 키 로드용

# 설치된 패키지에서 클래스 임포트
from clova_llama_index import ClovaClient, ClovaIndexEmbeddings, ClovaLLM

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.llms import ChatMessage, MessageRole

# --- 설정 ---
# API 키를 안전하게 로드합니다 (예: 환경 변수 사용)
# load_dotenv()
# clova_api_key = os.getenv("NAVER_CLOUD_CLOVA_API_KEY")
clova_api_key = "YOUR_NAVER_CLOUD_API_KEY" # 실제 네이버 클라우드 API 키로 교체하세요

if not clova_api_key or clova_api_key == "YOUR_NAVER_CLOUD_API_KEY":
    raise ValueError("NAVER_CLOUD_CLOVA_API_KEY 환경 변수를 설정하거나 코드 내 키를 입력해주세요.")

# --- 클라이언트 초기화 ---
# ClovaClient 인스턴스 생성
clova_client = ClovaClient(api_key=clova_api_key)
# 필요시 base_url 지정: clova_client = ClovaClient(api_key=clova_api_key, base_url="YOUR_API_GATEWAY_URL")

# --- 임베딩 사용 ---
# ClovaIndexEmbeddings 인스턴스 생성
clova_embedding = ClovaIndexEmbeddings(clova_client=clova_client)

# 예시: 쿼리에 대한 임베딩 얻기
try:
    query_vec = clova_embedding.get_query_embedding("HyperCLOVA X가 무엇인가요?")
    print(f"쿼리 임베딩 일부: {query_vec[:5]}...") # 첫 5개 요소 출력
except Exception as e:
    print(f"임베딩 생성 중 오류: {e}")

# --- LLM 사용 ---
# ClovaLLM 인스턴스 생성 (원하는 모델 지정)
clova_llm = ClovaLLM(client=clova_client, model="HCX-003") # HCX-003 외 다른 모델 사용 가능

# 예시: 간단한 텍스트 완성 (Completion)
try:
    prompt = "대규모 언어 모델(Large Language Models)에 대해 간단히 설명해주세요."
    completion_response = clova_llm.complete(prompt)
    print("\n--- Completion 응답 ---")
    print(completion_response.text)
except Exception as e:
    print(f"Completion 중 오류: {e}")

# 예시: 채팅 (Chat)
messages = [
    ChatMessage(role=MessageRole.SYSTEM, content="당신은 기술에 초점을 맞춘 도움이 되는 AI 어시스턴트입니다."),
    ChatMessage(role=MessageRole.USER, content="AI와 머신러닝의 차이점은 무엇인가요?"),
]
try:
    chat_response = clova_llm.chat(messages)
    print("\n--- Chat 응답 ---")
    print(chat_response.message.content)
except Exception as e:
    print(f"Chat 중 오류: {e}")


# --- LlamaIndex 와 함께 사용 (선택 사항) ---
# Settings를 사용하여 전역적으로 모델 설정 (권장 방식)
Settings.llm = clova_llm
Settings.embed_model = clova_embedding
# Settings.chunk_size = 512 # 필요시 청크 크기 등 설정

# # 예시: 로컬 디렉토리에서 문서를 로드하고 인덱스 생성
# # 테스트용 'data' 디렉토리 및 파일 생성
# if not os.path.exists("data"):
#      os.makedirs("data")
# with open("data/sample.txt", "w", encoding='utf-8') as f:
#      f.write("HyperCLOVA X는 네이버 클라우드의 초거대 AI 언어 모델입니다.")

# try:
#     documents = SimpleDirectoryReader("data").load_data()
#     index = VectorStoreIndex.from_documents(documents) # Settings에 설정된 임베딩 모델 사용
#     print("\n--- LlamaIndex 인덱스 생성 완료 (Clova 임베딩 사용) ---")

#     # 생성된 인덱스를 사용하여 쿼리 엔진 생성 (Settings에 설정된 LLM 사용)
#     query_engine = index.as_query_engine() # Settings.llm 사용
#     # query_engine = index.as_query_engine(llm=clova_llm) # 명시적으로 LLM 전달도 가능

#     response = query_engine.query("문서에 따르면 HyperCLOVA X는 무엇인가요?")
#     print("\n--- LlamaIndex 쿼리 엔진 응답 ---")
#     print(response)

# except Exception as e:
#      print(f"\nLlamaIndex 인덱싱 또는 쿼리 중 오류 발생 (data 디렉토리 및 파일 확인): {e}")
```