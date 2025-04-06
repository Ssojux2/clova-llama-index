import asyncio
from typing import List, Any
from pydantic import PrivateAttr, Field
from llama_index.core.embeddings import BaseEmbedding

# 패키지 내에서 상대 경로로 ClovaClient 임포트
from .client import ClovaClient

# 기본 배치 크기 상수화 (HyperCLOVA는 텍스트당 1개 호출)
DEFAULT_EMBED_BATCH_SIZE = 1

class ClovaIndexEmbeddings(BaseEmbedding):
    """
    네이버 클라우드 HyperCLOVA X Embedding API를 사용하는 LlamaIndex Embedding 클래스입니다.
    ClovaClient의 임베딩 기능을 LlamaIndex와 통합합니다.
    """
    # PrivateAttr: Pydantic 모델 외부에서 직접 접근/수정 방지 목적
    _client: ClovaClient = PrivateAttr()
    _model: str = PrivateAttr()
    # API는 텍스트별 처리지만, 호환성을 위해 유지
    # BaseEmbedding의 embed_batch_size를 사용하므로 PrivateAttr 불필요할 수 있음
    # _embed_batch_size: int = PrivateAttr()

    def __init__(
        self,
        clova_client: ClovaClient,
        model: str = "embedding_v2", # 추적/메타데이터용 모델 이름
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE, # 기본값 1로 설정
        **kwargs: Any,
    ) -> None:
        """
        ClovaIndexEmbeddings 어댑터를 초기화합니다.

        Args:
            clova_client: 초기화된 ClovaClient 인스턴스.
            model: 임베딩과 연관될 모델 이름.
            embed_batch_size: 임베딩 요청 배치 크기. 현재 API는 1이어야 합니다.
            **kwargs: BaseEmbedding 부모 클래스로 전달되는 추가 인수.
        """
        if embed_batch_size != DEFAULT_EMBED_BATCH_SIZE:
            print(f"경고: Clova Embedding API는 현재 배치 크기 {DEFAULT_EMBED_BATCH_SIZE}만 지원합니다. embed_batch_size를 {DEFAULT_EMBED_BATCH_SIZE}로 설정합니다.")
            embed_batch_size = DEFAULT_EMBED_BATCH_SIZE

        super().__init__(embed_batch_size=embed_batch_size, model_name=model, **kwargs) # model_name 전달

        self._client = clova_client
        self._model = model # 로컬에도 저장 (BaseEmbedding에도 저장됨)
        # self._embed_batch_size = embed_batch_size # BaseEmbedding의 속성 사용

    @classmethod
    def class_name(cls) -> str:
        """클래스 이름을 반환합니다."""
        return "ClovaIndexEmbeddings"

    # --- 비동기 메서드 ---
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """비동기적으로 쿼리 임베딩을 가져옵니다."""
        # 동기 네트워크 호출을 별도 스레드에서 실행하기 위해 asyncio.to_thread 사용
        try:
            res = await asyncio.to_thread(
                self._client.embeddings.create, input=query, model=self._model
            )
            data = res.get("data", [])
            if not data or not data[0].get("embedding"):
                # API 응답 형식에 따라 오류 처리 강화 가능
                raise ValueError("Clova API로부터 임베딩 데이터를 받지 못했습니다.")
            return data[0]["embedding"]
        except Exception as e:
            print(f"쿼리 임베딩 생성 중 오류 발생: {e}")
            raise # 또는 적절한 오류 처리

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """비동기적으로 텍스트 임베딩을 가져옵니다."""
        # Clova의 현재 API에서는 쿼리와 텍스트 임베딩이 동일한 엔드포인트 사용
        return await self._aget_query_embedding(text)

    # --- 동기 메서드 ---
    # LlamaIndex가 비동기 컨텍스트에서도 동기 메서드를 호출하거나 사용자가 직접 호출할 수 있음.

    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 임베딩을 가져옵니다."""
        # 클라이언트 헬퍼의 동기 메서드를 직접 호출
        try:
            res = self._client.embeddings.create(input=query, model=self._model)
            data = res.get("data", [])
            if not data or not data[0].get("embedding"):
                raise ValueError("Clova API로부터 임베딩 데이터를 받지 못했습니다.")
            return data[0]["embedding"]
        except Exception as e:
            print(f"쿼리 임베딩 생성 중 오류 발생: {e}")
            raise

    def _get_text_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩을 가져옵니다."""
        # Clova의 현재 API에서는 쿼리와 텍스트 임베딩이 동일한 엔드포인트 사용
        return self._get_query_embedding(text)

    # LlamaIndex >= 0.10 는 배치 처리를 위해 _get_text_embeddings/_aget_text_embeddings에 더 의존할 수 있음
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 목록에 대한 텍스트 임베딩을 가져옵니다."""
        # 배치 크기가 1이므로 텍스트를 순차적으로 처리합니다.
        # API 호출 사이에 지연(delay)이 client.embeddings.create에 구현되어 있음
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
         """비동기적으로 텍스트 목록에 대한 텍스트 임베딩을 가져옵니다."""
         # 배치 크기가 1이므로 비동기 메서드를 사용하여 텍스트를 순차적으로 처리합니다.
         # API가 배치를 허용한다면 asyncio.gather를 사용하여 잠재적 (제한적) 동시성 고려 가능
         embeddings = []
         for text in texts:
             emb = await self._aget_text_embedding(text)
             embeddings.append(emb)
         return embeddings

    # 선택 사항: 필요한 경우 get_query_embedding 구현 (기본 클래스가 처리할 수 있음)
    # def get_query_embedding(self, query: str) -> List[float]:
    #     return self._get_query_embedding(query)

    # 선택 사항: 직접 호출이 필요한 경우 __call__ 구현 (기본 클래스가 처리)
    # def __call__(self, texts: List[str]) -> List[List[float]]:
    #     return self._get_text_embeddings(texts)