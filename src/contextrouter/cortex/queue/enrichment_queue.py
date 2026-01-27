"""
Enrichment Queue - Redis-based shared queue for Gardener.

Used by:
- Worker scheduler: enqueue batches periodically
- Commerce AG-UI: enqueue single items on user action
- Gardener graph: dequeue and process

Features:
- Priority queue (high for user actions, low for scheduler)
- Deduplication (same product not processed twice)
- Batch state tracking
- Failed items retry
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class Priority(str, Enum):
    HIGH = "high"  # User-triggered (AG-UI)
    NORMAL = "normal"  # Scheduler batch
    LOW = "low"  # Retry failed


@dataclass
class QueueItem:
    """Item in enrichment queue."""

    product_id: int
    tenant_id: str
    priority: str = "normal"
    enqueued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "scheduler"  # scheduler, ag-ui, retry
    retry_count: int = 0  # Number of failed attempts

    MAX_RETRIES = 3  # After 3 fails → pending_human

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "QueueItem":
        d = json.loads(data)
        # Handle old items without retry_count
        if "retry_count" not in d:
            d["retry_count"] = 0
        return cls(**d)


@dataclass
class BatchState:
    """State of an enrichment batch."""

    batch_id: str
    product_ids: List[int]
    tenant_id: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "processing"  # processing, done, partial, failed
    processed_ids: List[int] = field(default_factory=list)
    failed_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "BatchState":
        return cls(**json.loads(data))


class EnrichmentQueue:
    """Redis-based shared queue for product enrichment.

    Queue structure:
    - enrichment:queue:{tenant}:high   - sorted set (priority queue)
    - enrichment:queue:{tenant}:normal
    - enrichment:queue:{tenant}:low
    - enrichment:processing:{product_id} - set of products being processed
    - enrichment:batch:{batch_id} - batch state
    """

    def __init__(self, redis_url: str, key_prefix: str = "enrichment:"):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self._redis = None

    async def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self.redis_url)
        return self._redis

    # --- Enqueue (Writer side: Worker/Commerce) ---

    async def enqueue(
        self,
        product_ids: List[int],
        tenant_id: str,
        priority: str = "normal",
        source: str = "scheduler",
    ) -> int:
        """Add products to enrichment queue.

        Args:
            product_ids: Products to enrich
            tenant_id: Tenant for multi-tenant
            priority: high (AG-UI), normal (scheduler), low (retry)
            source: Who enqueued (scheduler, ag-ui, retry)

        Returns:
            Number of items actually enqueued (excludes duplicates)
        """
        r = await self._get_redis()

        queue_key = f"{self.key_prefix}queue:{tenant_id}:{priority}"
        processing_key = f"{self.key_prefix}processing:{tenant_id}"

        enqueued = 0
        now = time.time()

        for pid in product_ids:
            # Skip if already processing
            if await r.sismember(processing_key, pid):
                logger.debug(f"Product {pid} already processing, skipping")
                continue

            # Skip if already in any queue
            for p in ["high", "normal", "low"]:
                if await r.zscore(f"{self.key_prefix}queue:{tenant_id}:{p}", pid):
                    logger.debug(f"Product {pid} already in queue, skipping")
                    continue

            # Add to queue with timestamp as score (FIFO within priority)
            item = QueueItem(
                product_id=pid,
                tenant_id=tenant_id,
                priority=priority,
                source=source,
            )

            # Store item data
            await r.set(f"{self.key_prefix}item:{pid}", item.to_json(), ex=3600)

            # Add to priority queue
            await r.zadd(queue_key, {str(pid): now})
            enqueued += 1

        if enqueued:
            logger.info(f"Enqueued {enqueued} products (priority={priority}, source={source})")

        return enqueued

    async def enqueue_one(
        self,
        product_id: int,
        tenant_id: str,
        priority: str = "high",  # Default high for single item (AG-UI)
    ) -> bool:
        """Enqueue single product (for AG-UI)."""
        return await self.enqueue([product_id], tenant_id, priority, source="ag-ui") == 1

    # --- Dequeue (Reader side: Gardener) ---

    async def dequeue(
        self,
        tenant_id: str,
        batch_size: int = 50,
        batch_id: str = None,
    ) -> List[int]:
        """Get next batch from queue for processing.

        Processes queues in priority order: high → normal → low

        Args:
            tenant_id: Tenant
            batch_size: Max items to dequeue
            batch_id: Optional batch ID for tracking

        Returns:
            List of product IDs to process
        """
        r = await self._get_redis()

        processing_key = f"{self.key_prefix}processing:{tenant_id}"
        product_ids = []

        # Process queues in priority order
        for priority in ["high", "normal", "low"]:
            if len(product_ids) >= batch_size:
                break

            queue_key = f"{self.key_prefix}queue:{tenant_id}:{priority}"
            needed = batch_size - len(product_ids)

            # Get oldest items (lowest score = oldest timestamp)
            items = await r.zrange(queue_key, 0, needed - 1)

            for item in items:
                pid = int(item)

                # Atomically move to processing
                removed = await r.zrem(queue_key, item)
                if removed:
                    await r.sadd(processing_key, pid)
                    product_ids.append(pid)

        if product_ids and batch_id:
            # Track batch state
            state = BatchState(
                batch_id=batch_id,
                product_ids=product_ids,
                tenant_id=tenant_id,
            )
            await r.setex(f"{self.key_prefix}batch:{batch_id}", 3600, state.to_json())

        if product_ids:
            logger.info(f"Dequeued {len(product_ids)} products for processing")

        return product_ids

    # --- Status updates (Gardener) ---

    async def mark_done(self, product_id: int, tenant_id: str, batch_id: str = None) -> None:
        """Mark product as successfully processed."""
        r = await self._get_redis()

        processing_key = f"{self.key_prefix}processing:{tenant_id}"
        await r.srem(processing_key, product_id)
        await r.delete(f"{self.key_prefix}item:{product_id}")

        if batch_id:
            await self._update_batch_state(batch_id, product_id, success=True)

    async def mark_failed(
        self,
        product_id: int,
        tenant_id: str,
        error: str = None,
        batch_id: str = None,
    ) -> str:
        """Mark product as failed.

        Automatically handles retry logic:
        - retry_count < 3: requeue with low priority
        - retry_count >= 3: return 'pending_human' for Commerce to handle

        Returns:
            'retrying' if requeued, 'pending_human' if max retries exceeded
        """
        r = await self._get_redis()

        processing_key = f"{self.key_prefix}processing:{tenant_id}"
        await r.srem(processing_key, product_id)

        if batch_id:
            await self._update_batch_state(batch_id, product_id, success=False, error=error)

        # Get current item to check retry count
        item_json = await r.get(f"{self.key_prefix}item:{product_id}")
        retry_count = 0
        if item_json:
            item = QueueItem.from_json(item_json.decode())
            retry_count = item.retry_count

        await r.delete(f"{self.key_prefix}item:{product_id}")

        if retry_count < QueueItem.MAX_RETRIES:
            # Requeue with incremented retry count
            new_item = QueueItem(
                product_id=product_id,
                tenant_id=tenant_id,
                priority="low",
                source="retry",
                retry_count=retry_count + 1,
            )
            await r.set(f"{self.key_prefix}item:{product_id}", new_item.to_json(), ex=3600)
            await r.zadd(f"{self.key_prefix}queue:{tenant_id}:low", {str(product_id): time.time()})

            logger.info(
                f"Product {product_id} requeued for retry ({retry_count + 1}/{QueueItem.MAX_RETRIES})"
            )
            return "retrying"
        else:
            # Max retries exceeded → needs human review
            logger.warning(f"Product {product_id} exceeded max retries, marking as pending_human")
            return "pending_human"

    async def complete_batch(self, batch_id: str) -> Optional[BatchState]:
        """Mark batch as complete and return final state."""
        r = await self._get_redis()

        state_json = await r.get(f"{self.key_prefix}batch:{batch_id}")
        if state_json:
            state = BatchState.from_json(state_json.decode())
            state.status = "done" if not state.failed_ids else "partial"
            await r.setex(f"{self.key_prefix}batch:{batch_id}", 3600, state.to_json())

            logger.info(
                f"Batch {batch_id} complete: "
                f"{len(state.processed_ids)} done, {len(state.failed_ids)} failed"
            )
            return state
        return None

    async def _update_batch_state(
        self,
        batch_id: str,
        product_id: int,
        success: bool,
        error: str = None,
    ) -> None:
        """Update batch state with product result."""
        r = await self._get_redis()

        state_json = await r.get(f"{self.key_prefix}batch:{batch_id}")
        if state_json:
            state = BatchState.from_json(state_json.decode())

            if success:
                if product_id not in state.processed_ids:
                    state.processed_ids.append(product_id)
            else:
                if product_id not in state.failed_ids:
                    state.failed_ids.append(product_id)
                if error:
                    state.error = error

            await r.setex(f"{self.key_prefix}batch:{batch_id}", 3600, state.to_json())

    # --- Query ---

    async def get_queue_stats(self, tenant_id: str) -> Dict[str, int]:
        """Get queue statistics."""
        r = await self._get_redis()

        stats = {}
        for priority in ["high", "normal", "low"]:
            queue_key = f"{self.key_prefix}queue:{tenant_id}:{priority}"
            stats[priority] = await r.zcard(queue_key)

        processing_key = f"{self.key_prefix}processing:{tenant_id}"
        stats["processing"] = await r.scard(processing_key)

        return stats

    async def is_processing(self, product_id: int, tenant_id: str) -> bool:
        """Check if product is currently being processed."""
        r = await self._get_redis()
        return await r.sismember(f"{self.key_prefix}processing:{tenant_id}", product_id)

    async def get_batch(self, batch_id: str) -> Optional[BatchState]:
        """Get batch state."""
        r = await self._get_redis()
        state_json = await r.get(f"{self.key_prefix}batch:{batch_id}")
        if state_json:
            return BatchState.from_json(state_json.decode())
        return None

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


# Singleton for shared access
_queue: Optional[EnrichmentQueue] = None


def get_enrichment_queue(redis_url: str = None) -> EnrichmentQueue:
    """Get or create shared enrichment queue."""
    global _queue

    if _queue is None:
        if redis_url is None:
            from ...core.config import get_core_config

            config = get_core_config()
            redis_url = config.redis.url

        _queue = EnrichmentQueue(redis_url)

    return _queue


async def close_enrichment_queue():
    """Close shared queue connection."""
    global _queue
    if _queue:
        await _queue.close()
        _queue = None
