"""
Content Ingestion API Router
Endpoints for ingesting SELVE framework content, blog posts, and external sources
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.services.content_ingestion_service import ContentIngestionService


router = APIRouter(prefix="/api/ingestion", tags=["ingestion"])


class ContentItem(BaseModel):
    """Single content item for ingestion"""
    content: str
    source: str
    content_type: str
    metadata: Optional[Dict[str, Any]] = None
    validate: bool = True


class BatchIngestionRequest(BaseModel):
    """Batch ingestion request"""
    contents: List[ContentItem]


class IngestionResponse(BaseModel):
    """Response from ingestion operation"""
    ingested: bool
    chunks_created: Optional[int] = None
    embedding_cost: Optional[float] = None
    validation_status: Optional[str] = None
    content_hash: Optional[str] = None
    sync_record_id: Optional[str] = None
    error: Optional[str] = None


class BatchIngestionResponse(BaseModel):
    """Response from batch ingestion"""
    total: int
    successful: int
    failed: int
    total_cost: float
    results: List[Dict[str, Any]]


class IngestionStatsResponse(BaseModel):
    """Statistics about ingested content"""
    total_syncs: int
    total_chunks: int
    total_cost: float
    by_source: Dict[str, Dict[str, Any]]
    recent_syncs: List[Dict[str, Any]]


ingestion_service = ContentIngestionService()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_content(item: ContentItem):
    """
    Ingest a single content item

    Chunks content, generates embeddings, and stores in Qdrant vector database.
    """
    try:
        result = await ingestion_service.ingest_content(
            content=item.content,
            source=item.source,
            content_type=item.content_type,
            metadata=item.metadata,
            validate=item.validate
        )

        return IngestionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/ingest/batch", response_model=BatchIngestionResponse)
async def ingest_batch(request: BatchIngestionRequest):
    """
    Ingest multiple content items in batch

    More efficient than individual ingestion for large datasets.
    """
    try:
        # Convert Pydantic models to dicts
        contents = [
            {
                "content": item.content,
                "source": item.source,
                "content_type": item.content_type,
                "metadata": item.metadata,
                "validate": item.validate
            }
            for item in request.contents
        ]

        result = await ingestion_service.ingest_batch(contents)

        return BatchIngestionResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch ingestion failed: {str(e)}"
        )


@router.get("/stats", response_model=IngestionStatsResponse)
async def get_ingestion_stats():
    """
    Get statistics about ingested content

    Returns counts, costs, and recent ingestion operations.
    """
    try:
        stats = await ingestion_service.get_ingestion_stats()

        return IngestionStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}"
        )


@router.post("/ingest/file")
async def ingest_from_file(
    file_path: str,
    source: str,
    content_type: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Ingest content from a file

    Supports .txt, .md files. Reads file and ingests content.
    """
    try:
        import os

        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File not found: {file_path}"
            )

        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add filename to metadata
        file_metadata = metadata or {}
        file_metadata["filename"] = os.path.basename(file_path)
        file_metadata["file_path"] = file_path

        # Ingest
        result = await ingestion_service.ingest_content(
            content=content,
            source=source,
            content_type=content_type,
            metadata=file_metadata,
            validate=True
        )

        return IngestionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(e)}"
        )


@router.post("/ingest/directory")
async def ingest_from_directory(
    directory_path: str,
    source: str,
    content_type: str,
    file_extensions: List[str] = [".txt", ".md"],
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Ingest all files from a directory

    Recursively scans directory and ingests all matching files.
    """
    try:
        import os
        import glob

        if not os.path.exists(directory_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Directory not found: {directory_path}"
            )

        # Find all matching files
        all_files = []
        for ext in file_extensions:
            pattern = os.path.join(directory_path, f"**/*{ext}")
            all_files.extend(glob.glob(pattern, recursive=True))

        if not all_files:
            return {
                "message": "No files found",
                "total": 0,
                "successful": 0,
                "failed": 0
            }

        # Ingest each file
        contents = []
        for file_path in all_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                file_metadata = metadata.copy() if metadata else {}
                file_metadata["filename"] = os.path.basename(file_path)
                file_metadata["file_path"] = file_path

                contents.append({
                    "content": content,
                    "source": source,
                    "content_type": content_type,
                    "metadata": file_metadata,
                    "validate": True
                })

            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")

        # Batch ingest
        result = await ingestion_service.ingest_batch(contents)

        return BatchIngestionResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Directory ingestion failed: {str(e)}"
        )
