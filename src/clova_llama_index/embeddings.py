import asyncio
from typing import List, Any
from pydantic import PrivateAttr
import time

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

    def __init__(
        self,
        clova_client: ClovaClient,
        model: str = "embedding_v2",  # 추적/메타데이터용 모델 이름
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,  # 기본값 1로 설정
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

        super().__init__(embed_batch_size=embed_batch_size, model_name=model, **kwargs)  # model_name 전달
        self._client = clova_client
        self._model = model  # 로컬에도 저장 (BaseEmbedding에도 저장됨)

    @classmethod
    def class_name(cls) -> str:
        """클래스 이름을 반환합니다."""
        return "ClovaIndexEmbeddings"

    # --- 비동기 메서드 ---
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """비동기적으로 쿼리 임베딩을 가져옵니다."""
        try:
            res = await asyncio.to_thread(
                self._client.embeddings.create, input=query, model=self._model
            )
            data = res.get("data", [])
            if not data or not data[0].get("embedding"):
                raise ValueError("Clova API로부터 임베딩 데이터를 받지 못했습니다.")
            return data[0]["embedding"]
        except Exception as e:
            print(f"쿼리 임베딩 생성 중 오류 발생: {e}")
            raise

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """비동기적으로 텍스트 임베딩을 가져옵니다."""
        return await self._aget_query_embedding(text)

    # --- 동기 메서드 ---
    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 임베딩을 가져옵니다."""
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
        return self._get_query_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """텍스트 목록에 대한 텍스트 임베딩을 가져옵니다."""
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """비동기적으로 텍스트 목록에 대한 텍스트 임베딩을 가져옵니다."""
        embeddings = []
        for text in texts:
            emb = await self._aget_text_embedding(text)
            embeddings.append(emb)
        return embeddings

    # 수정 사항: chromadb EmbeddingFunction 인터페이스에 맞게 __call__ 메서드 정의
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        임베딩 함수 호출 시, chromadb는 매개변수 이름을 'input'으로 기대합니다.
        이 메서드는 입력 텍스트 목록에 대해 텍스트 임베딩을 동기적으로 반환합니다.
        """
        return self._get_text_embeddings(input)
